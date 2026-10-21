"""Streamlit chat UI.  Run:  streamlit run app.py"""
import os

import streamlit as st

from analytics import InvoiceData
from chatbot import InvoiceChatbot
from llm import AnthropicLLM, LLMError, OpenAICompatLLM
from report import build_report

st.set_page_config(page_title="Invoice Analyst", page_icon="🧾", layout="wide")


@st.cache_data
def load_data():
    return InvoiceData("data")


data = load_data()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Model")
    provider = st.selectbox("Provider", ["Offline (no LLM)", "Anthropic (Claude)", "Ollama / OpenAI-compatible"])
    llm, status = None, "Offline mode: keyword routing, templated answers."
    try:
        if provider.startswith("Anthropic"):
            model = st.text_input("Model", "claude-sonnet-5-5")
            key = st.text_input("API key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
            if key:
                llm, status = AnthropicLLM(model, key), f"Using {model}"
            else:
                status = "Enter an API key to enable the LLM."
        elif provider.startswith("Ollama"):
            model = st.text_input("Model", "llama3.1:8b")
            url = st.text_input("Base URL", "http://localhost:11434/v1")
            llm, status = OpenAICompatLLM(model, url), f"Using {model} at {url}"
    except LLMError as e:
        status = f"LLM unavailable: {e}"
    st.caption(status)

    st.divider()
    st.header("Dataset")
    st.metric("Invoices", len(data.inv))
    st.metric("Total billed", data.money(data.inv.total.sum()))
    with st.expander("Preview data"):
        st.dataframe(data.inv.drop(columns=["month", "quarter"]), use_container_width=True, height=300)

    st.divider()
    st.header("Report")
    if st.button("Generate full report", use_container_width=True):
        with st.spinner("Building report..."):
            path = build_report(data, "reports/invoice_report.pdf", llm)
        st.session_state["report_bytes"] = path.read_bytes()
    if "report_bytes" in st.session_state:
        st.download_button("Download PDF report", st.session_state["report_bytes"],
                           "invoice_report.pdf", "application/pdf", use_container_width=True)

# ---------------- chat ----------------
st.title("🧾 Invoice Analyst")
st.caption("Ask about spend, vendors, trends, overdue invoices, anomalies - and request charts.")

bot = InvoiceChatbot(data, llm)
st.session_state.setdefault("messages", [])      # rendered turns (with tables/figures)

SUGGESTIONS = ["Give me a summary of all invoices", "Show spend by category as a pie chart",
               "Monthly spend trend", "Which invoices are overdue?", "Top 5 vendors by total",
               "Any anomalies or duplicate invoices?", "Which vendors do we pay late?"]
cols = st.columns(len(SUGGESTIONS[:4]))
clicked = None
for c, q in zip(cols, SUGGESTIONS[:4]):
    if c.button(q, use_container_width=True):
        clicked = q
with st.expander("More example questions"):
    st.write("\n".join(f"- {q}" for q in SUGGESTIONS[4:]) +
             "\n- Show Cloud & Hosting spend by month as a line chart\n- List invoices above 100000 that are unpaid")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for f in m.get("figures", []):
            st.pyplot(f)
        for t in m.get("tables", []):
            st.dataframe(t, use_container_width=True)

prompt = st.chat_input("Ask about your invoices...") or clicked
if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            ans = bot.ask(prompt, history)
        st.markdown(ans.text)
        st.caption("🟢 Answered with LLM" if ans.mode == "llm" else "🟠 Offline fallback (LLM not used or unreachable)")
        for f in ans.figures:
            st.pyplot(f)
        for t in ans.tables:
            st.dataframe(t, use_container_width=True)
        if ans.calls:
            with st.expander("How this was computed"):
                st.json(ans.calls)
    st.session_state.messages.append({"role": "assistant", "content": ans.text,
                                      "figures": ans.figures, "tables": ans.tables})
