# Invoice Analyst Chatbot

Chat with 100 invoices: ask questions, get charts, download a PDF financial report.

## Run
```bash
pip install -r requirements.txt
python generate_invoices.py        # creates data/ (100 PDFs + CSVs)  -- already included
python tests/test_smoke.py         # sanity checks, no API key needed
streamlit run app.py
```
In the sidebar choose a model: **Anthropic** (paste API key), **Ollama** (free, local, e.g. `ollama pull llama3.1:8b`),
or **Offline** (keyword routing, no LLM). "Generate full report" builds `reports/invoice_report.pdf`.

## How it works
```
question -> LLM planner (picks tools + args as JSON) -> pandas tools compute -> LLM narrates the verified results
                                       \-> chart (matplotlib) + table shown in chat
```
The LLM never does arithmetic, so every number is traceable (see "How this was computed" under each answer).
If the LLM is unavailable or returns bad JSON, a keyword router keeps the app working.

| File | Role |
|---|---|
| `generate_invoices.py` | 100 synthetic invoices as PDFs + `invoices.csv` + `line_items.csv` |
| `analytics.py` | Tools: KPIs, group-by, trends, overdue ageing, payment performance, anomalies, growth |
| `chatbot.py` / `llm.py` | Planner + narrator, provider wrappers |
| `report.py` | PDF report: summary, KPIs, charts, risks, recommendations |
| `app.py` | Streamlit chat UI |

## Using your own invoices
Replace `data/invoices.csv` and `data/line_items.csv` (and set `as_of`/`currency` in `data/meta.json`).
`invoices.csv` columns: invoice_id, vendor, category, invoice_date, due_date, payment_terms_days, status (Paid|Pending|Overdue),
paid_date, payment_method, subtotal, discount, tax_rate, tax, total, currency.
`line_items.csv`: invoice_id, description, quantity, unit_price, amount.
To get these from PDFs, add an extraction step (PyMuPDF/pdfplumber text, or have the LLM return this JSON per invoice).
