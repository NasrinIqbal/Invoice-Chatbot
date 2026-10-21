import os
import pandas as pd
from sqlalchemy import create_engine

url = os.getenv("DATABASE_URL")
print("DATABASE_URL set:", bool(url))
engine = create_engine(url)
print(pd.read_sql("SELECT COUNT(*) AS invoices, SUM(total) AS total_billed FROM invoices", engine))