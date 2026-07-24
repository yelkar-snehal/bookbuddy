import polars as pl

df = pl.read_parquet("data/processed/Goodreads-Books.parquet")

print(df.columns)
print(df.head(5))