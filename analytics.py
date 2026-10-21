"""Deterministic analytics layer.

The LLM never does arithmetic. It picks one of the tools below and the tools
compute every number with pandas, so figures in answers are verifiable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PALETTE = ["#1F3A5F", "#3E7CB1", "#81A4CD", "#F4A259", "#5B8E7D", "#BC4B51", "#8D6A9F", "#C9B037"]
DIMENSIONS = ["vendor", "category", "status", "payment_method", "month", "quarter"]
METRICS = ["total", "count", "tax", "subtotal", "discount", "average"]
FILTER_KEYS = ["vendor", "category", "status", "date_from", "date_to"]


@dataclass
class Result:
    text: str
    table: Optional[pd.DataFrame] = None
    figure: Optional[plt.Figure] = None
    error: bool = False


class InvoiceData:
    def __init__(self, data_dir: str | Path = "data"):
        d = Path(data_dir)
        meta = json.loads((d / "meta.json").read_text())
        self.currency = meta.get("currency", "INR")
        self.as_of = pd.Timestamp(meta["as_of"])
        date_cols = ["invoice_date", "due_date", "paid_date"]
        self.inv = pd.read_csv(d / "invoices.csv", parse_dates=date_cols)
        self.items = pd.read_csv(d / "line_items.csv")
        inv = self.inv
        inv["month"] = inv["invoice_date"].dt.to_period("M").astype(str)
        inv["quarter"] = inv["invoice_date"].dt.to_period("Q").astype(str)
        inv["days_to_pay"] = (inv["paid_date"] - inv["invoice_date"]).dt.days
        inv["days_late"] = (inv["paid_date"] - inv["due_date"]).dt.days
        inv["days_overdue"] = ((self.as_of - inv["due_date"]).dt.days).where(inv["status"] == "Overdue")

    # ---------- helpers ----------
    def money(self, v: float) -> str:
        return f"{self.currency} {v:,.0f}"

    def _filter(self, **f) -> pd.DataFrame:
        df = self.inv
        for key in ("vendor", "category", "status"):
            if f.get(key):
                df = df[df[key].str.lower().str.contains(str(f[key]).lower(), regex=False)]
        if f.get("date_from"):
            df = df[df["invoice_date"] >= pd.Timestamp(f["date_from"])]
        if f.get("date_to"):
            df = df[df["invoice_date"] <= pd.Timestamp(f["date_to"])]
        return df

    def _agg(self, df: pd.DataFrame, dim: str, metric: str) -> pd.DataFrame:
        g = df.groupby(dim)
        if metric == "count":
            out = g.size().rename("count")
        elif metric == "average":
            out = g["total"].mean().rename("average")
        else:
            out = g[metric].sum().rename(metric)
        out = out.reset_index()
        if dim in ("month", "quarter"):
            return out.sort_values(dim)
        return out.sort_values(out.columns[-1], ascending=False)

    @staticmethod
    def _chart(df, x, y, kind, title, ylabel="", money=True):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        labels, vals = df[x].astype(str).tolist(), df[y].tolist()
        if kind == "pie":
            ax.pie(vals, labels=labels, autopct="%1.0f%%", colors=PALETTE, startangle=90,
                   wedgeprops={"linewidth": 1, "edgecolor": "white"})
            ax.axis("equal")
        elif kind == "line":
            ax.plot(labels, vals, marker="o", color=PALETTE[0], linewidth=2)
            if len(vals) >= 4:
                roll = pd.Series(vals).rolling(3).mean()
                ax.plot(labels, roll, linestyle="--", color=PALETTE[3], label="3-period average")
                ax.legend(frameon=False)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", alpha=0.25)
        elif kind == "barh":
            ax.barh(labels[::-1], vals[::-1], color=PALETTE[0])
            ax.grid(axis="x", alpha=0.25)
        else:
            ax.bar(labels, vals, color=PALETTE[0])
            ax.tick_params(axis="x", rotation=30)
            ax.grid(axis="y", alpha=0.25)
        if kind != "pie":
            ax.set_ylabel(ylabel or y)
            if money:
                ax.yaxis.set_major_formatter(lambda v, _: f"{v/1000:,.0f}k" if abs(v) >= 1000 else f"{v:,.0f}")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
        fig.tight_layout()
        return fig

    def _empty(self) -> Result:
        return Result("No invoices match those filters.", error=True)

    # ---------- tools ----------
    def kpi_summary(self, **f) -> Result:
        df = self._filter(**f)
        if df.empty:
            return self._empty()
        paid = df[df.status == "Paid"]
        unpaid = df[df.status != "Paid"]
        kpis = {
            "Invoices": len(df), "Total billed": self.money(df.total.sum()),
            "Average invoice": self.money(df.total.mean()), "Largest invoice": self.money(df.total.max()),
            "Total tax": self.money(df.tax.sum()), "Total discounts": self.money(df.discount.sum()),
            "Paid": self.money(paid.total.sum()), "Outstanding (pending + overdue)": self.money(unpaid.total.sum()),
            "Overdue": self.money(df[df.status == "Overdue"].total.sum()),
            "Avg days to pay": f"{paid.days_to_pay.mean():.1f}" if len(paid) else "n/a",
            "Period": f"{df.invoice_date.min():%d %b %Y} to {df.invoice_date.max():%d %b %Y}",
        }
        tbl = pd.DataFrame({"Metric": kpis.keys(), "Value": [str(v) for v in kpis.values()]})
        return Result("Key figures: " + "; ".join(f"{k}: {v}" for k, v in kpis.items()), tbl)

    def group_by(self, dimension: str = "vendor", metric: str = "total", top_n: Optional[int] = None,
                 chart: Optional[str] = None, **f) -> Result:
        if dimension not in DIMENSIONS or metric not in METRICS:
            return Result(f"dimension must be one of {DIMENSIONS} and metric one of {METRICS}.", error=True)
        df = self._filter(**f)
        if df.empty:
            return self._empty()
        out = self._agg(df, dimension, metric)
        if top_n:
            out = out.head(int(top_n))
        col = out.columns[-1]
        share = out[col] / out[col].sum() * 100 if metric in ("total", "count", "tax") else None
        shown = out.copy()
        if share is not None:
            shown["share_%"] = share.round(1)
        fig = None
        if chart in ("bar", "barh", "pie", "line"):
            kind = chart
            if dimension in ("month", "quarter") and chart == "pie":
                kind = "bar"
            fig = self._chart(out, dimension, col, kind, f"{col.title()} by {dimension}",
                              money=metric != "count")
        lead = out.iloc[0]
        text = (f"{col} by {dimension}: top is {lead[dimension]} at "
                f"{lead[col]:,.0f}" + (f" ({share.iloc[0]:.1f}% of the shown total)" if share is not None else "") + ".")
        return Result(text, shown, fig)

    def _is_partial(self, period: str, freq: str) -> bool:
        p = pd.Period(period, freq="M" if freq == "month" else "Q")
        return p.end_time.normalize() > self.as_of

    def trend(self, metric: str = "total", freq: str = "month", chart: bool = True, **f) -> Result:
        freq = freq if freq in ("month", "quarter") else "month"
        df = self._filter(**f)
        if df.empty:
            return self._empty()
        out = self._agg(df, freq, metric if metric in METRICS else "total")
        col = out.columns[-1]
        out["partial"] = [self._is_partial(p, freq) for p in out[freq]]
        out["change_%"] = (out[col].pct_change() * 100).round(1)
        out.loc[out["partial"], "change_%"] = float("nan")   # an unfinished period is not comparable
        done = out[~out["partial"]]
        peak = done.loc[done[col].idxmax()]
        last = done.iloc[-1]
        text = (f"{freq.title()}ly {col}: peak {peak[freq]} ({peak[col]:,.0f}); last complete {freq} {last[freq]} "
                f"{last[col]:,.0f} ({last['change_%']}% vs prior).")
        if out["partial"].any():
            text += f" The latest {freq} ({out.iloc[-1][freq]}) is still in progress, so it is excluded from comparisons."
        plot = out.drop(columns=["partial"])
        fig = self._chart(plot, freq, col, "line", f"{freq.title()}ly {col} trend" + (" (last period incomplete)" if out["partial"].any() else ""),
                          money=metric != "count") if chart else None
        return Result(text, out.drop(columns=["partial"]), fig)

    def overdue_report(self, **f) -> Result:
        df = self._filter(**f)
        od = df[df.status == "Overdue"].sort_values("days_overdue", ascending=False)
        if od.empty:
            return Result("There are no overdue invoices.")
        bins = pd.cut(od.days_overdue, [0, 30, 60, 90, 10_000], labels=["1-30", "31-60", "61-90", "90+"])
        aging = od.groupby(bins, observed=False).agg(invoices=("invoice_id", "count"), amount=("total", "sum")).reset_index()
        aging.columns = ["days_overdue_bucket", "invoices", "amount"]
        fig = self._chart(aging, "days_overdue_bucket", "amount", "bar", "Overdue amount by age (days)")
        top = od.groupby("vendor").total.sum().sort_values(ascending=False)
        text = (f"{len(od)} invoices are overdue, worth {self.money(od.total.sum())}. Oldest is "
                f"{od.iloc[0].invoice_id} ({od.iloc[0].vendor}), {int(od.iloc[0].days_overdue)} days late. "
                f"Largest overdue vendor: {top.index[0]} ({self.money(top.iloc[0])}).")
        tbl = od[["invoice_id", "vendor", "due_date", "days_overdue", "total"]].copy()
        tbl["due_date"] = tbl["due_date"].dt.date
        return Result(text, tbl, fig)

    def payment_performance(self, **f) -> Result:
        df = self._filter(**f)
        paid = df[df.status == "Paid"]
        if paid.empty:
            return self._empty()
        g = paid.groupby("vendor").agg(invoices=("invoice_id", "count"), avg_days_to_pay=("days_to_pay", "mean"),
                                       late_rate=("days_late", lambda s: (s > 0).mean() * 100),
                                       avg_days_late=("days_late", lambda s: s[s > 0].mean())).round(1)
        g = g.reset_index().sort_values("late_rate", ascending=False)
        fig = self._chart(g, "vendor", "late_rate", "barh", "Share of invoices paid late (%)", "%", money=False)
        overall = (paid.days_late > 0).mean() * 100
        text = (f"{overall:.0f}% of paid invoices were settled after the due date; average time to pay "
                f"{paid.days_to_pay.mean():.1f} days. Worst vendor for late payment: {g.iloc[0].vendor} "
                f"({g.iloc[0].late_rate:.0f}%).")
        return Result(text, g, fig)

    def find_invoices(self, min_total: Optional[float] = None, max_total: Optional[float] = None,
                      sort_by: str = "total", limit: int = 15, **f) -> Result:
        df = self._filter(**f)
        if min_total is not None:
            df = df[df.total >= float(min_total)]
        if max_total is not None:
            df = df[df.total <= float(max_total)]
        if df.empty:
            return self._empty()
        sort_by = sort_by if sort_by in ("total", "invoice_date", "due_date") else "total"
        df = df.sort_values(sort_by, ascending=False)
        tbl = df[["invoice_id", "vendor", "category", "invoice_date", "due_date", "status", "total"]].head(int(limit)).copy()
        tbl["invoice_date"], tbl["due_date"] = tbl["invoice_date"].dt.date, tbl["due_date"].dt.date
        return Result(f"{len(df)} invoices match (showing {len(tbl)}), totalling {self.money(df.total.sum())}.", tbl)

    def top_line_items(self, n: int = 10, **f) -> Result:
        ids = self._filter(**f).invoice_id
        it = self.items[self.items.invoice_id.isin(ids)]
        if it.empty:
            return self._empty()
        g = it.groupby("description").agg(times_billed=("invoice_id", "count"), amount=("amount", "sum")) \
              .sort_values("amount", ascending=False).head(int(n)).reset_index()
        fig = self._chart(g, "description", "amount", "barh", "Top line items by spend")
        return Result(f"Biggest spend item: {g.iloc[0].description} ({self.money(g.iloc[0].amount)}, pre-tax).", g, fig)

    def anomalies(self, **f) -> Result:
        df = self._filter(**f).copy()
        if df.empty:
            return self._empty()
        stats = df.groupby("category").total.agg(["median", "std"]).rename(columns={"median": "cat_median", "std": "cat_std"})
        df = df.join(stats, on="category")
        df["z"] = (df.total - df.cat_median) / df.cat_std.replace(0, pd.NA)
        outliers = df[df.z > 2.5][["invoice_id", "vendor", "category", "total", "cat_median", "z"]].copy()
        outliers["flag"] = "Unusually high for category"
        key = df.groupby(["vendor", "total"]).invoice_id.transform("count")
        dups = df[key > 1].sort_values(["vendor", "total"])[["invoice_id", "vendor", "category", "total", "cat_median", "z"]].copy()
        dups["flag"] = "Possible duplicate (same vendor and amount)"
        out = pd.concat([outliers, dups]).round({"cat_median": 0, "z": 1})
        if out.empty:
            return Result("No anomalies detected.")
        return Result(f"Found {len(outliers)} unusually high invoices and {len(dups)} rows that look like duplicates.", out)

    def category_growth(self) -> Result:
        """Last 3 COMPLETE months vs the 3 months before, per category."""
        df = self.inv
        last_m = self.as_of.to_period("M") - (0 if self.as_of == self.as_of.to_period("M").end_time.normalize() else 1)
        m = df.invoice_date.dt.to_period("M")
        recent, prior = df[(m > last_m - 3) & (m <= last_m)], df[(m > last_m - 6) & (m <= last_m - 3)]
        t = pd.DataFrame({"recent_3m": recent.groupby("category").total.sum(), "prior_3m": prior.groupby("category").total.sum(),
                          "recent_n": recent.groupby("category").size(), "prior_n": prior.groupby("category").size()}).fillna(0)
        t["change_%"] = ((t.recent_3m / t.prior_3m.where(t.prior_3m > 0) - 1) * 100).astype(float).round(1)
        t = t.reset_index().sort_values("change_%", ascending=False)
        t[["recent_n", "prior_n"]] = t[["recent_n", "prior_n"]].astype(int)
        valid = t[(t.prior_n >= 2)].dropna(subset=["change_%"])
        hi, lo = valid.iloc[0], valid.iloc[-1]
        return Result(f"Comparing {last_m - 2} to {last_m} with the 3 months before: biggest rise (categories with 2+ prior invoices) {hi.category} ({hi['change_%']}%), "
                      f"biggest fall {lo.category} ({lo['change_%']}%). Invoice counts per category are small, so treat swings with care.", t)


TOOLS = {
    "kpi_summary": (InvoiceData.kpi_summary, "Headline KPIs: count, total billed, average, tax, paid, outstanding, overdue."),
    "group_by": (InvoiceData.group_by, "Aggregate by dimension. args: dimension(vendor|category|status|payment_method|month|quarter), "
                                       "metric(total|count|tax|subtotal|discount|average), top_n, chart(bar|barh|pie|line or null)."),
    "trend": (InvoiceData.trend, "Time trend. args: metric(total|count|tax...), freq(month|quarter), chart(true/false)."),
    "overdue_report": (InvoiceData.overdue_report, "Overdue invoices with ageing buckets."),
    "payment_performance": (InvoiceData.payment_performance, "Days-to-pay and late-payment rate per vendor."),
    "find_invoices": (InvoiceData.find_invoices, "List invoices. args: min_total, max_total, sort_by(total|invoice_date|due_date), limit."),
    "top_line_items": (InvoiceData.top_line_items, "Top products/services by spend. args: n."),
    "anomalies": (InvoiceData.anomalies, "Unusually large invoices and possible duplicates."),
    "category_growth": (InvoiceData.category_growth, "Which categories grew or shrank (last 3 months vs previous 3)."),
}
FILTER_DOC = "Every tool also accepts optional filters: vendor, category, status (Paid|Pending|Overdue), date_from, date_to (YYYY-MM-DD)."


def run_tool(data: InvoiceData, name: str, args: dict | None = None) -> Result:
    if name not in TOOLS:
        return Result(f"Unknown tool '{name}'.", error=True)
    fn = TOOLS[name][0]
    args = {k: v for k, v in (args or {}).items() if v not in (None, "", "null")}
    try:
        return fn(data, **args)
    except TypeError as e:
        return Result(f"Bad arguments for {name}: {e}", error=True)
    except Exception as e:  # never let a bad plan crash the chat
        return Result(f"Could not run {name}: {e}", error=True)
