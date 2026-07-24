import polars as pl

df = pl.read_parquet("data/processed/Goodreads-Books.parquet")

print(df.columns)
print(df.head(5))

row = df.row(0, named=True)

for key, value in row.items():

    print(f"\n=== {key} ===")

    print(value)

print(df["community_reviews"][0][:2000])

print(df.select([
    pl.col("summary").is_null().sum().alias("missing_summary"),
    pl.col("genres").is_null().sum().alias("missing_genres"),
    pl.col("author").is_null().sum().alias("missing_author"),
]))

print(

    df.select(

        [

            (pl.col("genres") == "").sum().alias("empty_genres"),

            (pl.col("summary") == "").sum().alias("empty_summaries"),

            (pl.col("author") == "").sum().alias("empty_authors"),

        ]

    )

)