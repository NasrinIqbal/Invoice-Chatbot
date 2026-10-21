"""Generate a reproducible dataset of 100 realistic invoices.

Outputs
  data/invoices/*.pdf   - one PDF per invoice (what you would receive in real life)
  data/invoices.csv     - one row per invoice (header level)
  data/line_items.csv   - one row per invoice line
  data/meta.json        - as-of date + currency

To use your OWN invoices instead, keep the same two CSV schemas (see README)
or write an extractor that produces them from your PDFs.
"""
import json
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SEED = 42
N_INVOICES = 100
AS_OF = date(2026, 10, 16)
START = date(2025, 10, 1)
END = date(2026, 10, 10)
CURRENCY = "INR"
BILL_TO = "Acme Technologies Pvt Ltd, Kochi, Kerala"

# vendor -> (category, weight, tax_rate, [(description, low, high)])
VENDORS = {
    "CloudNimbus Hosting": ("Cloud & Hosting", 12, 0.18, [
        ("Compute instances (monthly)", 38000, 52000), ("Object storage", 9000, 15000),
        ("Managed database", 21000, 30000), ("Data transfer", 4000, 9000)]),
    "DevStack Software": ("Software Subscriptions", 10, 0.18, [
        ("Team licence - IDE", 14000, 20000), ("CI/CD minutes", 6000, 11000),
        ("Issue tracker seats", 8000, 12000), ("Security scanning", 12000, 18000)]),
    "OfficeMart Supplies": ("Office Supplies", 14, 0.12, [
        ("A4 paper (cartons)", 1800, 3200), ("Printer toner", 3500, 7000),
        ("Stationery pack", 1200, 2600), ("Pantry supplies", 2000, 4500)]),
    "TechVista Hardware": ("IT Hardware", 7, 0.18, [
        ("Laptop - 14in", 62000, 88000), ("Monitor 27in", 14000, 22000),
        ("Docking station", 7000, 11000), ("Network switch", 18000, 30000)]),
    "BrightReach Marketing": ("Marketing", 9, 0.18, [
        ("Social ad campaign", 25000, 60000), ("Content writing", 12000, 24000),
        ("SEO retainer", 18000, 30000), ("Event sponsorship", 40000, 90000)]),
    "SwiftLine Logistics": ("Logistics", 11, 0.05, [
        ("Courier - domestic", 3000, 9000), ("Freight - air", 15000, 42000),
        ("Warehousing (monthly)", 20000, 34000)]),
    "Kerala Power & Utilities": ("Utilities", 12, 0.0, [
        ("Electricity - office", 14000, 24000), ("Internet - leased line", 9000, 13000),
        ("Water & maintenance", 2500, 5000)]),
    "LexAssure Consulting": ("Professional Services", 8, 0.18, [
        ("Legal retainer", 30000, 55000), ("Statutory audit support", 45000, 80000),
        ("Compliance review", 20000, 38000)]),
    "PixelForge Design": ("Marketing", 4, 0.18, [
        ("Brand design sprint", 35000, 60000), ("UI illustration pack", 12000, 22000)]),
    "SecureNet Solutions": ("IT Hardware", 5, 0.18, [
        ("Firewall appliance", 55000, 85000), ("Endpoint protection licences", 16000, 26000)]),
}
METHODS = ["Bank Transfer", "UPI", "Credit Card", "Cheque"]


def build() -> None:
    rnd = random.Random(SEED)
    names = list(VENDORS)
    weights = [VENDORS[n][1] for n in names]
    # ~8 invoices per month (Oct-2025..Sep-2026) + the first days of Oct-2026, so no month is empty
    counts = [7, 9, 8, 8, 8, 9, 8, 7, 8, 8, 9, 7]
    dates = []
    for k, n in enumerate(counts):
        y, m = divmod(START.month - 1 + k, 12)
        first = date(START.year + y, m + 1, 1)
        nxt = date(first.year + (m == 11), (m + 1) % 12 + 1, 1)
        dates += [first + timedelta(days=rnd.randrange((nxt - first).days)) for _ in range(n)]
    dates += [date(2026, 10, rnd.randint(1, 10)) for _ in range(N_INVOICES - len(dates))]
    dates.sort()

    invoices, lines = [], []
    for i, inv_date in enumerate(dates, start=1):
        vendor = rnd.choices(names, weights)[0]
        cat, _, tax_rate, catalog = VENDORS[vendor]
        month_idx = (inv_date.year - START.year) * 12 + inv_date.month - START.month

        n_lines = rnd.randint(1, min(3, len(catalog)))
        picks = rnd.sample(catalog, n_lines)
        # planted trends: cloud bills creep up ~3.5%/month; hardware spikes in Mar & Sep
        growth = 1 + 0.035 * month_idx if cat == "Cloud & Hosting" else 1.0
        spike = 1.35 if (cat == "IT Hardware" and inv_date.month in (3, 9)) else 1.0
        subtotal = 0.0
        inv_id = f"INV-{inv_date.year}-{i:04d}"
        for desc, lo, hi in picks:
            qty = rnd.randint(1, 4) if cat in ("IT Hardware", "Office Supplies") else 1
            unit = round(rnd.uniform(lo, hi) * growth * spike, -1)
            amount = qty * unit
            subtotal += amount
            lines.append(dict(invoice_id=inv_id, description=desc, quantity=qty,
                              unit_price=unit, amount=amount))
        # two planted outliers for the anomaly detector
        if i in (37, 81):
            for ln in lines[-n_lines:]:
                ln["unit_price"] *= 4.2
                ln["amount"] *= 4.2
            subtotal *= 4.2
        discount = round(subtotal * rnd.choice([0, 0, 0, 0.02, 0.05]), 0)
        tax = round((subtotal - discount) * tax_rate, 0)
        total = subtotal - discount + tax

        terms = rnd.choice([15, 30, 30, 45])
        due = inv_date + timedelta(days=terms)
        if due < AS_OF:
            if rnd.random() < 0.82:
                status = "Paid"
                paid_date = due + timedelta(days=rnd.randint(-8, 6 if vendor != "SwiftLine Logistics" else 25))
            else:
                status, paid_date = "Overdue", None
        else:
            status = "Paid" if rnd.random() < 0.3 else "Pending"
            paid_date = inv_date + timedelta(days=rnd.randint(3, max(4, (AS_OF - inv_date).days))) if status == "Paid" else None
        if paid_date and paid_date > AS_OF:
            paid_date = AS_OF

        invoices.append(dict(
            invoice_id=inv_id, vendor=vendor, category=cat, invoice_date=inv_date,
            due_date=due, payment_terms_days=terms, status=status, paid_date=paid_date,
            payment_method=rnd.choice(METHODS) if status == "Paid" else None,
            subtotal=subtotal, discount=discount, tax_rate=tax_rate, tax=tax, total=total,
            currency=CURRENCY))

    # a planted duplicate invoice (same vendor, same amount, 2 days apart)
    src = invoices[60].copy()
    dup_id = "INV-2026-0101"
    src.update(invoice_id=dup_id, invoice_date=src["invoice_date"] + timedelta(days=2))
    src["due_date"] = src["invoice_date"] + timedelta(days=src["payment_terms_days"])
    src["status"], src["paid_date"], src["payment_method"] = "Pending" if src["due_date"] >= AS_OF else "Overdue", None, None
    invoices.append(src)
    for ln in [l for l in lines if l["invoice_id"] == invoices[60]["invoice_id"]]:
        lines.append({**ln, "invoice_id": dup_id})
    # trim back to exactly 100 by dropping the first small invoice (keeps ids unique)
    drop = invoices.pop(0)["invoice_id"]
    lines = [l for l in lines if l["invoice_id"] != drop]

    out = Path("data")
    (out / "invoices").mkdir(parents=True, exist_ok=True)
    inv_df, line_df = pd.DataFrame(invoices), pd.DataFrame(lines)
    inv_df.to_csv(out / "invoices.csv", index=False)
    line_df.to_csv(out / "line_items.csv", index=False)
    (out / "meta.json").write_text(json.dumps({"as_of": AS_OF.isoformat(), "currency": CURRENCY}))
    for inv in invoices:
        write_pdf(inv, [l for l in lines if l["invoice_id"] == inv["invoice_id"]], out / "invoices")
    print(f"Wrote {len(inv_df)} invoices, {len(line_df)} line items, {len(inv_df)} PDFs -> {out}/")


def write_pdf(inv: dict, items: list, folder: Path) -> None:
    st = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(folder / f"{inv['invoice_id']}.pdf"), pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm)
    fmt = lambda v: f"{v:,.2f}"
    story = [Paragraph(f"<b>{inv['vendor']}</b>", st["Title"]),
             Paragraph(f"TAX INVOICE &nbsp; | &nbsp; {inv['invoice_id']}", st["Heading3"]),
             Paragraph(f"Bill to: {BILL_TO}", st["Normal"]),
             Paragraph(f"Invoice date: {inv['invoice_date']} &nbsp; Due date: {inv['due_date']} "
                       f"&nbsp; Terms: Net {inv['payment_terms_days']}", st["Normal"]),
             Paragraph(f"Category: {inv['category']}", st["Normal"]), Spacer(1, 8 * mm)]
    rows = [["Description", "Qty", f"Unit ({inv['currency']})", f"Amount ({inv['currency']})"]]
    rows += [[l["description"], l["quantity"], fmt(l["unit_price"]), fmt(l["amount"])] for l in items]
    rows += [["", "", "Subtotal", fmt(inv["subtotal"])], ["", "", "Discount", f"-{fmt(inv['discount'])}"],
             ["", "", f"Tax ({inv['tax_rate']:.0%})", fmt(inv["tax"])], ["", "", "TOTAL", fmt(inv["total"])]]
    t = Table(rows, colWidths=[85 * mm, 15 * mm, 35 * mm, 35 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                           ("LINEBELOW", (0, 0), (-1, len(items)), 0.25, colors.grey),
                           ("FONTNAME", (2, -1), (-1, -1), "Helvetica-Bold")]))
    story += [t, Spacer(1, 8 * mm), Paragraph(f"Payment status: <b>{inv['status']}</b>", st["Normal"])]
    doc.build(story)


if __name__ == "__main__":
    build()
