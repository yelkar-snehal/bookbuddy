from pathlib import Path

import polars as pl

Path("data/processed").mkdir(parents=True, exist_ok=True)

df = pl.scan_csv("data/raw/Goodreads-Books.csv")
df.collect().write_parquet("data/processed/Goodreads-Books.parquet")
