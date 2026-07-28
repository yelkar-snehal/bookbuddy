import polars as pl
import random

# ---------- Load ----------
df = pl.read_parquet("data/processed/Goodreads-Books.parquet")

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)

print(f"Rows: {df.height:,}")
print(f"Columns: {df.width}")
print()

print("Columns:")
for col, dtype in zip(df.columns, df.dtypes):
    print(f"  - {col}: {dtype}")

# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("MISSING / EMPTY VALUES")
print("=" * 80)

for col in df.columns:
    nulls = df[col].is_null().sum()

    if df[col].dtype == pl.String:
        empties = (df[col] == "").sum()
    else:
        empties = "-"

    print(f"{col:<20} null={nulls:<8} empty={empties}")

# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("DUPLICATES")
print("=" * 80)

print("Duplicate IDs:",
      df["id"].is_duplicated().sum())

print("Duplicate URLs:",
      df["url"].is_duplicated().sum())

print("Duplicate Title + Author:",
      df.select(["name", "author"])
        .is_duplicated()
        .sum())

# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("SUMMARY LENGTH")
print("=" * 80)

summary_lengths = (
    df
    .with_columns(
        pl.col("summary")
        .str.len_chars()
        .alias("summary_length")
    )
)

print(
    summary_lengths
    .select("summary_length")
    .describe()
)

# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("RATING DISTRIBUTION")
print("=" * 80)

print(
    df
    .select(
        pl.col("num_ratings")
        .cast(pl.Int64, strict=False)
    )
    .describe()
)

# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("AUTHOR EXAMPLES")
print("=" * 80)

authors = (
    df["author"]
    .drop_nulls()
    .unique()
    .sample(10, shuffle=True)
)

for author in authors:
    print(author)

# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("GENRE EXAMPLES")
print("=" * 80)

genres = (
    df
    .filter(pl.col("genres") != "")
    .select("genres")
    .sample(10)
)

print(genres)

# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("ABOUT AUTHOR EXAMPLES")
print("=" * 80)

authors = (
    df
    .select("about_author")
    .sample(5)
)

print(authors)

# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("COMMUNITY REVIEW EXAMPLES")
print("=" * 80)

reviews = (
    df
    .select("community_reviews")
    .sample(5)
)

print(reviews)

# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("RANDOM BOOKS")
print("=" * 80)

sample = df.sample(5)

for row in sample.iter_rows(named=True):
    print("-" * 80)
    print(f"Title   : {row['name']}")
    print(f"Author  : {row['author']}")
    print(f"Rating  : {row['star_rating']} ({row['num_ratings']} ratings)")
    print(f"Genres  : {row['genres']}")
    print(f"Summary : {row['summary'][:250]}...")