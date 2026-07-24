import polars as pl

df = pl.scan_csv("data/raw/Goodreads-Books.csv")
df.collect().write_parquet("data/raw/Goodreads-Books.parquet")