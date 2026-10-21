import os
import pandas as pd
from sqlalchemy import create_engine

url = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:Password%402255@localhost:5432/invoices")
engine = create_engine(url)

dates = ["invoice_date", "due_date", "paid_date"]
pd.read_csv("data/invoices.csv", parse_dates=dates).to_sql("invoices", engine, if_exists="replace", index=False)
pd.read_csv("data/line_items.csv").to_sql("line_items", engine, if_exists="replace", index=False)
print("Loaded into PostgreSQL")