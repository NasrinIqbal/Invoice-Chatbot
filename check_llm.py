from analytics import InvoiceData
from chatbot import InvoiceChatbot
from llm import OpenAICompatLLM, LLMError

llm = OpenAICompatLLM("llama3.1:8b", "http://localhost:11434/v1")

print("TEST 1: is Ollama reachable?")
try:
    print(llm.complete("Reply with JSON only.", [{"role": "user", "content": 'Return {"ok": true}'}]))
except LLMError as e:
    print("FAILED:", e)
    raise SystemExit

print("\nTEST 2: what does the model plan?")
bot = InvoiceChatbot(InvoiceData("data"), llm)
try:
    print(bot._plan("show overdue invoices above 100000", []))
except LLMError as e:
    print("FAILED:", e)