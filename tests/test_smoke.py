"""Run: python tests/test_smoke.py   (no API key needed; the LLM is faked)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analytics import InvoiceData, run_tool
from chatbot import InvoiceChatbot, keyword_router
from report import build_report


class FakeLLM:
    """Plays the planner (returns JSON) and the narrator (returns prose)."""

    def __init__(self, plan):
        self.plan = plan

    def complete(self, system, messages, max_tokens=1000):
        return self.plan if "planning step" in system else "Narrated answer based on the tool results."


def main():
    d = InvoiceData("data")
    assert len(d.inv) == 100 and d.inv.invoice_id.is_unique
    assert abs((d.inv.total - (d.inv.subtotal - d.inv.discount + d.inv.tax)).abs().max()) < 1, "totals must reconcile"
    line_sum = d.items.groupby("invoice_id").amount.sum().reindex(d.inv.invoice_id).values
    assert abs(line_sum - d.inv.subtotal.values).max() < 1, "line items must sum to subtotal"

    r = run_tool(d, "group_by", {"dimension": "category", "chart": "pie"})
    assert r.figure is not None and abs(r.table["total"].sum() - d.inv.total.sum()) < 1
    assert run_tool(d, "group_by", {"dimension": "nope"}).error
    assert run_tool(d, "no_such_tool").error

    plan = json.dumps({"calls": [{"tool": "group_by", "args": {"dimension": "vendor", "top_n": 3, "chart": "bar"}}]})
    a = InvoiceChatbot(d, FakeLLM("```json\n" + plan + "\n```")).ask("top 3 vendors chart")
    assert a.mode == "llm" and len(a.tables[0]) == 3 and a.figures

    a = InvoiceChatbot(d, FakeLLM("not json at all")).ask("show overdue invoices")  # malformed -> fallback
    assert a.mode == "offline" and a.calls[0]["tool"] == "overdue_report"
    assert InvoiceChatbot(d, None).ask("spend by category pie").figures
    assert keyword_router("any duplicate invoices?")[0]["tool"] == "anomalies"

    p = build_report(d, "reports/invoice_report.pdf", FakeLLM(""))
    assert p.stat().st_size > 50_000
    print("all checks passed")


if __name__ == "__main__":
    main()
