"""Invoice chatbot: LLM plans -> pandas tools compute -> LLM explains."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from analytics import FILTER_DOC, TOOLS, InvoiceData, Result, run_tool
from llm import LLMError

MAX_CALLS = 3

PLANNER_SYSTEM = """You are the planning step of an invoice-analytics assistant.
Choose which analysis tools to run to answer the user's latest question. You cannot see the data.

TOOLS:
{tools}
{filters}

Vendors: {vendors}
Categories: {categories}
Data covers {start} to {end}. Today is {today}. Currency: {currency}.

Reply with ONLY a JSON object, no prose, in one of these forms:
{{"calls": [{{"tool": "<name>", "args": {{...}}}}]}}      (1-{max_calls} calls)
{{"calls": [], "reply": "<short answer>"}}               (greeting, or the question is unrelated to the invoices)

Rules: use group_by with chart set when the user asks for a chart/graph/plot/pie/bar; use chart "line" for time
trends; resolve follow-ups ("now as a pie", "just for Q2") from the conversation; dates must be YYYY-MM-DD."""

ANSWER_SYSTEM = """You are a precise financial analyst assistant. Answer the user's question using ONLY the tool
results provided. Never invent or recalculate numbers; quote them as given. Be concise (under 150 words), lead with
the answer, mention notable patterns, and if a chart or table is attached say so briefly. If the results show an
error or no data, say so plainly and suggest a rephrasing."""


@dataclass
class Answer:
    text: str
    tables: list[pd.DataFrame] = field(default_factory=list)
    figures: list = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    mode: str = "llm"


def _extract_json(raw: str) -> Optional[dict]:
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    start = raw.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(raw)):
            depth += (raw[i] == "{") - (raw[i] == "}")
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    break
        start = raw.find("{", start + 1)
    return None


def keyword_router(q: str) -> list[dict]:
    """Offline fallback so the app is usable (and testable) without any LLM."""
    q = q.lower()
    chart = "pie" if "pie" in q else "line" if "line" in q else "bar" if re.search(r"chart|graph|plot|bar|visual", q) else None
    if re.search(r"anomal|duplicate|unusual|outlier|suspicious", q):
        return [{"tool": "anomalies", "args": {}}]
    if "overdue" in q or "late" in q and "pay" not in q:
        return [{"tool": "overdue_report", "args": {}}]
    if re.search(r"pay(ment)? (performance|speed|time)|days to pay|late pay|paid late", q):
        return [{"tool": "payment_performance", "args": {}}]
    if re.search(r"trend|over time|monthly|month by month|quarter", q):
        return [{"tool": "trend", "args": {"freq": "quarter" if "quarter" in q else "month"}}]
    if re.search(r"grow|rising|increas", q):
        return [{"tool": "category_growth", "args": {}}]
    if re.search(r"line item|product|service", q):
        return [{"tool": "top_line_items", "args": {}}]
    dim = ("category" if "categor" in q else "status" if "status" in q else "payment_method" if re.search(r"method|mode", q)
           else "vendor" if re.search(r"vendor|supplier|who", q) else None)
    if dim:
        metric = "count" if re.search(r"how many|count|number", q) else "total"
        return [{"tool": "group_by", "args": {"dimension": dim, "metric": metric, "chart": chart or "bar"}}]
    if re.search(r"list|show|find|largest|biggest|invoices", q):
        return [{"tool": "find_invoices", "args": {}}]
    return [{"tool": "kpi_summary", "args": {}}]


class InvoiceChatbot:
    def __init__(self, data: InvoiceData, llm=None):
        self.data, self.llm = data, llm

    def _planner_prompt(self) -> str:
        d = self.data
        tools = "\n".join(f"- {n}: {desc}" for n, (_, desc) in TOOLS.items())
        return PLANNER_SYSTEM.format(
            tools=tools, filters=FILTER_DOC, vendors=", ".join(sorted(d.inv.vendor.unique())),
            categories=", ".join(sorted(d.inv.category.unique())), start=f"{d.inv.invoice_date.min():%Y-%m-%d}",
            end=f"{d.inv.invoice_date.max():%Y-%m-%d}", today=f"{d.as_of:%Y-%m-%d}", currency=d.currency,
            max_calls=MAX_CALLS)

    def _plan(self, question: str, history: list[dict]) -> tuple[list[dict], Optional[str]]:
        msgs = [*history[-6:], {"role": "user", "content": question}]
        for _ in range(2):  # one retry if the model returns malformed JSON
            parsed = _extract_json(self.llm.complete(self._planner_prompt(), msgs, max_tokens=500))
            if parsed is not None:
                calls = [c for c in parsed.get("calls", []) if isinstance(c, dict) and c.get("tool") in TOOLS]
                return calls[:MAX_CALLS], parsed.get("reply")
            msgs = [*msgs, {"role": "user", "content": "Return ONLY valid JSON in the required format."}]
        raise LLMError("planner returned invalid JSON")

    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        history = history or []
        mode, reply = "llm", None
        try:
            if self.llm is None:
                raise LLMError("offline")
            calls, reply = self._plan(question, history)
        except LLMError:
            mode, calls = "offline", keyword_router(question)
        if not calls:
            return Answer(reply or "I can answer questions about the invoices - try asking for a summary.", mode=mode)

        results: list[Result] = [run_tool(self.data, c["tool"], c.get("args")) for c in calls]
        tables = [r.table for r in results if r.table is not None]
        figures = [r.figure for r in results if r.figure is not None]
        facts = "\n\n".join(
            f"[{c['tool']} {json.dumps(c.get('args', {}))}]\n{r.text}" + (
                f"\nTable (first 12 rows):\n{r.table.head(12).to_csv(index=False)}" if r.table is not None else "")
            for c, r in zip(calls, results))
        text = " ".join(r.text for r in results)
        if mode == "llm":
            try:
                msg = [*history[-4:], {"role": "user", "content": f"Question: {question}\n\nTool results:\n{facts}"}]
                text = self.llm.complete(ANSWER_SYSTEM, msg, max_tokens=500).strip()
            except LLMError:
                mode = "offline"  # keep the computed facts even if narration fails
        return Answer(text, tables, figures, calls, mode)
