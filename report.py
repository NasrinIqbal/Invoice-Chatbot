"""Comprehensive PDF report. Facts are computed in pandas; the LLM (optional) only writes the narrative."""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from analytics import InvoiceData, run_tool
from llm import LLMError

NAVY = colors.HexColor("#1F3A5F")


def compute_insights(d: InvoiceData) -> dict:
    """Return {'findings': [...], 'risks': [...], 'recommendations': [...]} from hard numbers."""
    inv, m = d.inv, d.money
    paid = inv[inv.status == "Paid"]
    overdue = inv[inv.status == "Overdue"]
    findings, risks, recs = [], [], []

    by_v = inv.groupby("vendor").total.sum().sort_values(ascending=False)
    by_c = inv.groupby("category").total.sum().sort_values(ascending=False)
    findings.append(f"Total billed was {m(inv.total.sum())} across {len(inv)} invoices from {inv.vendor.nunique()} vendors "
                    f"(average {m(inv.total.mean())}, median {m(inv.total.median())}).")
    findings.append(f"{by_c.index[0]} is the largest category at {by_c.iloc[0] / by_c.sum():.0%} of spend; "
                    f"{by_v.index[0]} is the largest vendor at {by_v.iloc[0] / by_v.sum():.0%}.")
    top3 = by_v.head(3).sum() / by_v.sum()
    findings.append(f"The top 3 vendors account for {top3:.0%} of spend.")
    if top3 > 0.5:
        risks.append(f"Vendor concentration: three vendors hold {top3:.0%} of spend.")
        recs.append("Review contracts with the top vendors and benchmark pricing against alternatives to reduce concentration risk.")

    monthly = inv.groupby("month").total.sum()
    complete = [p for p in monthly.index if not d._is_partial(p, "month")]
    if len(complete) >= 6:
        recent, prior = monthly[complete[-3:]].sum(), monthly[complete[-6:-3]].sum()
        chg = (recent / prior - 1) * 100
        findings.append(f"Spend in the last three complete months ({complete[-3]} to {complete[-1]}) was {m(recent)}, "
                        f"{chg:+.1f}% versus the prior three months.")
        findings.append(f"Peak month was {monthly[complete].idxmax()} ({m(monthly[complete].max())}); "
                        f"lowest was {monthly[complete].idxmin()} ({m(monthly[complete].min())}).")

    cloud = inv[inv.category == "Cloud & Hosting"].groupby("month").total.sum()
    cloud = cloud[[p for p in cloud.index if not d._is_partial(p, "month")]]
    if len(cloud) >= 6 and cloud.iloc[-3:].mean() > cloud.iloc[:3].mean() * 1.3:
        findings.append(f"Cloud & Hosting bills are creeping up: average monthly spend rose from {m(cloud.iloc[:3].mean())} "
                        f"(first 3 months) to {m(cloud.iloc[-3:].mean())} (last 3).")
        recs.append("Audit cloud usage (right-size instances, reserved pricing, storage lifecycle rules); spend is trending upward.")

    if len(overdue):
        share = overdue.total.sum() / inv.total.sum()
        risks.append(f"{len(overdue)} invoices worth {m(overdue.total.sum())} ({share:.0%} of billed value) are overdue; "
                     f"{int((overdue.days_overdue > 90).sum())} are more than 90 days late.")
        recs.append("Clear or formally dispute invoices overdue by 90+ days to avoid late fees and supplier-relationship damage.")
    late = (paid.days_late > 0)
    if len(paid):
        findings.append(f"{late.mean():.0%} of paid invoices were settled after their due date; average time to pay was "
                        f"{paid.days_to_pay.mean():.1f} days.")
        worst = paid.groupby("vendor").days_late.apply(lambda s: (s > 0).mean())
        if worst.max() > 0.5:
            recs.append(f"Tighten the payment run for {worst.idxmax()} (paid late on {worst.max():.0%} of invoices).")
    findings.append(f"Tax charged totals {m(inv.tax.sum())} ({inv.tax.sum() / inv.subtotal.sum():.1%} of subtotal); "
                    f"discounts captured total {m(inv.discount.sum())} ({inv.discount.sum() / inv.subtotal.sum():.1%}).")
    if inv.discount.sum() / inv.subtotal.sum() < 0.03:
        recs.append("Negotiate early-payment discounts; only a small share of spend currently earns one.")

    an = run_tool(d, "anomalies")
    if an.table is not None:
        dup = an.table[an.table.flag.str.startswith("Possible")]
        high = an.table[an.table.flag.str.startswith("Unusually")]
        if len(high):
            risks.append(f"{len(high)} invoice(s) are unusually large for their category: {', '.join(high.invoice_id)}.")
        if len(dup):
            risks.append(f"Possible duplicate billing: {', '.join(dup.invoice_id)} share vendor and amount.")
            recs.append("Verify the flagged duplicate and unusually high invoices before payment.")
    return {"findings": findings, "risks": risks, "recommendations": recs}


def _narrative(llm, ins: dict, kpis: str) -> str | None:
    if llm is None:
        return None
    prompt = ("Write a 120-word executive summary for a CFO from these verified facts. Use only these numbers.\n\n"
              + kpis + "\nFindings:\n- " + "\n- ".join(ins["findings"]) + "\nRisks:\n- " + "\n- ".join(ins["risks"]))
    try:
        return llm.complete("You are a concise financial analyst.", [{"role": "user", "content": prompt}], 400).strip()
    except LLMError:
        return None


def _img(fig, width_mm=170):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    buf.seek(0)
    w, h = fig.get_size_inches()
    return Image(buf, width=width_mm * mm, height=width_mm * mm * h / w)


def _table(df: pd.DataFrame, widths=None, max_rows=25):
    df = df.head(max_rows).copy()
    rows = [list(df.columns)] + [[f"{v:,.1f}" if isinstance(v, float) else str(v) for v in r] for r in df.itertuples(index=False)]
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), 8), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F9")]),
                           ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")), ("ALIGN", (1, 0), (-1, -1), "RIGHT")]))
    return t


def build_report(d: InvoiceData, path: str | Path = "reports/invoice_report.pdf", llm=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=NAVY, spaceBefore=10)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=NAVY, spaceBefore=8)
    body = ParagraphStyle("B", parent=ss["BodyText"], leading=14)
    bullet = ParagraphStyle("Bul", parent=body, leftIndent=12, bulletIndent=2)

    ins = compute_insights(d)
    kpi = run_tool(d, "kpi_summary")
    kpi_text = "KPIs: " + kpi.text
    summary = _narrative(llm, ins, kpi_text)
    story = [Paragraph("Invoice Financial Analysis Report", ParagraphStyle("T", parent=ss["Title"], textColor=NAVY)),
             Paragraph(f"{len(d.inv)} invoices | {d.inv.invoice_date.min():%d %b %Y} - {d.inv.invoice_date.max():%d %b %Y} | "
                       f"data as of {d.as_of:%d %b %Y} | generated {datetime.now():%d %b %Y}", ss["Italic"]), Spacer(1, 6 * mm),
             Paragraph("1. Executive summary", H1)]
    if summary:
        story += [Paragraph(summary, body), Spacer(1, 3 * mm)]
    story += [Paragraph(f, bullet, bulletText="\u2022") for f in ins["findings"][:4]]
    story += [Paragraph("2. Key performance indicators", H1), _table(kpi.table, [70 * mm, 100 * mm])]

    story += [Spacer(1, 4 * mm), Paragraph("3. Spending analysis", H1)]
    for tool, args, caption in [
        ("trend", {}, "Monthly spend. The dashed line is a 3-month moving average; the final month is incomplete."),
        ("group_by", {"dimension": "category", "chart": "pie"}, "Spend by category."),
        ("group_by", {"dimension": "vendor", "chart": "barh", "top_n": 10}, "Top vendors by billed value."),
        ("top_line_items", {"n": 8}, "Highest-spend products and services (pre-tax).")]:
        r = run_tool(d, tool, args)
        story += [_img(r.figure, 125), Paragraph(f"<i>{caption}</i> {r.text}", body), Spacer(1, 4 * mm)]
    cg = run_tool(d, "category_growth")
    story += [Paragraph("Category momentum", H2), Paragraph(cg.text, body), _table(cg.table.round(1))]

    story += [PageBreak(), Paragraph("4. Payments and cash-flow risk", H1)]
    status = run_tool(d, "group_by", {"dimension": "status", "chart": "pie"})
    od, pp = run_tool(d, "overdue_report"), run_tool(d, "payment_performance")
    story += [_img(status.figure, 85), Paragraph(status.text, body), Spacer(1, 3 * mm)]
    if od.figure:
        story += [_img(od.figure, 120), Paragraph(od.text, body), Spacer(1, 3 * mm)]
    story += [_img(pp.figure, 125), Paragraph(pp.text, body)]

    story += [Paragraph("5. Anomalies and control checks", H1)]
    an = run_tool(d, "anomalies")
    story.append(Paragraph(an.text, body))
    if an.table is not None:
        story += [Spacer(1, 3 * mm), _table(an.table.drop(columns=["cat_median", "z"], errors="ignore"))]

    story += [Paragraph("6. Key findings", H1)] + [Paragraph(f, bullet, bulletText="\u2022") for f in ins["findings"]]
    story += [Paragraph("7. Risks", H1)] + [Paragraph(f, bullet, bulletText="\u2022") for f in ins["risks"] or ["No material risks detected."]]
    story += [Paragraph("8. Recommendations", H1)] + [Paragraph(f, bullet, bulletText=f"{i}.") for i, f in enumerate(ins["recommendations"], 1)]
    story += [Spacer(1, 6 * mm), Paragraph("<i>Method: all figures are computed directly from the invoice data with pandas. "
              "Anomalies use a per-category z-score above 2.5 and same-vendor same-amount matching.</i>", ss["Italic"])]

    def footer(c, doc):
        c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.grey)
        c.drawRightString(A4[0] - 20 * mm, 10 * mm, f"Page {doc.page}"); c.restoreState()

    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm,
                      bottomMargin=18 * mm, title="Invoice Financial Analysis Report").build(story, onFirstPage=footer, onLaterPages=footer)
    return path
