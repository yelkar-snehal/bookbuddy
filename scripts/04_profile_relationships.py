import json
import polars as pl

df = pl.read_parquet("data/processed/Goodreads-Books.parquet")

print("=" * 80)
print("MULTIPLE AUTHORS")
print("=" * 80)


def author_count(author_str: str) -> int:
    if not author_str:
        return 0
    try:
        return len(json.loads(author_str))
    except Exception:
        return -1


author_counts = (
    df.select(
        pl.col("author")
        .map_elements(author_count, return_dtype=pl.Int64)
        .alias("author_count")
    )
)

print(author_counts.group_by("author_count").len().sort("author_count"))

# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("DUPLICATE IDS")
print("=" * 80)

duplicate_ids = (
    df.group_by("id")
    .len()
    .filter(pl.col("len") > 1)
    .head(5)
)

print(duplicate_ids)

for row in duplicate_ids.iter_rows(named=True):
    print("\n" + "-" * 80)
    print(f"ID: {row['id']}")

    books = (
        df.filter(pl.col("id") == row["id"])
        .select(
            [
                "name",
                "author",
                "url",
                "first_published",
                "star_rating",
            ]
        )
    )

    print(books)

# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("ABOUT AUTHOR SCHEMA")
print("=" * 80)

samples = (
    df.select("about_author")
    .sample(1000)
    .to_series()
)

keys = {}

for item in samples:
    try:
        obj = json.loads(item)
        for key in obj.keys():
            keys[key] = keys.get(key, 0) + 1
    except Exception:
        pass

for key, count in sorted(keys.items(), key=lambda x: -x[1]):
    print(f"{key:<30} {count}")

# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FIRST PUBLISHED EXAMPLES")
print("=" * 80)

(
    df.select("first_published")
    .unique()
    .sample(20)
    .sort("first_published")
    .write_csv("/dev/stdout")
)

# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("PUBLICATION DATE PATTERNS")
print("=" * 80)

print("Contains '/' :",
      df.filter(pl.col("first_published").str.contains("/")).height)

print("Length == 4 :",
      df.filter(pl.col("first_published").str.len_chars() == 4).height)

print("Empty :",
      df.filter(pl.col("first_published") == "").height)