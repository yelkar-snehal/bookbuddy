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


print("\n" + "=" * 80)
print("UNIQUE AUTHORS")
print("=" * 80)

unique_authors = (
    df
    .filter(pl.col("author") != "")
    .select("author")
    .n_unique()
)

print(f"Unique author values: {unique_authors:,}")


print("\n" + "=" * 80)
print("AUTHOR METADATA CONSISTENCY")
print("=" * 80)

# Find authors that appear across multiple books.
# For each author, check how many distinct about_author values they have.

author_metadata = (
    df
    .filter(pl.col("author") != "")
    .group_by("author")
    .agg(
        pl.len().alias("book_count"),
        pl.col("about_author").n_unique().alias("about_author_variants"),
    )
    .sort("book_count", descending=True)
)

print("Authors with multiple books:")
print(
    author_metadata
    .filter(pl.col("book_count") > 1)
    .head(10)
)

print("\nAuthors with inconsistent about_author metadata:")
print(
    author_metadata
    .filter(pl.col("about_author_variants") > 1)
    .head(10)
)


print("\n" + "=" * 80)
print("DUPLICATE ID INVESTIGATION")
print("=" * 80)

duplicate_id_stats = (
    df
    .group_by("id")
    .agg(
        pl.len().alias("row_count"),
        pl.col("url").n_unique().alias("unique_urls"),
        pl.col("name").n_unique().alias("unique_titles"),
        pl.col("author").n_unique().alias("unique_authors"),
    )
    .filter(pl.col("row_count") > 1)
)

print(
    duplicate_id_stats
    .group_by(["unique_urls", "unique_titles", "unique_authors"])
    .len()
    .sort("len", descending=True)
)

print("\nExamples where duplicate IDs have different URLs:")

different_url_ids = (
    duplicate_id_stats
    .filter(pl.col("unique_urls") > 1)
    .head(5)
)

print(different_url_ids)

for row in different_url_ids.iter_rows(named=True):
    book_id = (
        df
        .filter(pl.col("id") == row["id"])
        .select(["id", "name", "author", "url"])
    )

    print("\n" + "-" * 80)
    print(book_id)