from decimal import Decimal
import json

import polars as pl
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import engine
from app.db.models import Author, Book


PARQUET_PATH = "data/processed/Goodreads-Books.parquet"
SAMPLE_SIZE = None


def parse_kindle_price(value: str) -> Decimal | None:
    if not value:
        return None

    return Decimal(value.replace('"', "").replace("$", "").replace(",", "").strip())


def parse_author(value: str) -> str | None:
    if not value:
        return None

    authors = json.loads(value)

    if not authors:
        return None

    return authors[0]


def parse_author_metadata(value: str) -> dict:
    if not value:
        return {}

    metadata = json.loads(value)
    num_followers = metadata.get("num_followers")

    return {
        "num_books": metadata.get("num_books"),
         "num_followers": int(num_followers) if num_followers else None,
        "about": metadata.get("about"),
    }


def transform_row(row: dict) -> dict:
    return {
        "goodreads_id": row["id"],
        "url": row["url"],
        "title": row["name"],
        "author_name": parse_author(row["author"]),
        "author_metadata": parse_author_metadata(row["about_author"]),
        "summary": row["summary"] or None,
        "star_rating": row["star_rating"],
        "num_ratings": row["num_ratings"],
        "num_reviews": (
            int(row["num_reviews"])
            if row["num_reviews"]
            else None
        ),
        "genres": (
            json.loads(row["genres"])
            if row["genres"]
            else None
        ),
        "first_published": row["first_published"] or None,
        "kindle_price": parse_kindle_price(row["kindle_price"]),
        "community_reviews": (
            json.loads(row["community_reviews"])
            if row["community_reviews"]
            else None
        ),
    }

def ingest_batch(session: Session, rows: list[dict]) -> dict:
    author_names = {
        row["author_name"]
        for row in rows
        if row["author_name"]
    }

    existing_authors = session.scalars(
        select(Author).where(Author.name.in_(author_names))
    ).all()

    authors_by_name = {
        author.name: author
        for author in existing_authors
    }

    authors_created = 0

    for row in rows:
        author_name = row["author_name"]

        if not author_name or author_name in authors_by_name:
            continue

        metadata = row["author_metadata"]

        author = Author(
            name=author_name,
            num_books=metadata["num_books"],
            num_followers=metadata["num_followers"],
            about=metadata["about"],
        )

        session.add(author)
        authors_by_name[author_name] = author
        authors_created += 1

    session.flush()

    book_urls = {row["url"] for row in rows}

    existing_urls = set(
        session.scalars(
            select(Book.url).where(Book.url.in_(book_urls))
        ).all()
    )

    books = []

    for row in rows:
        if row["url"] in existing_urls:
            continue

        author = authors_by_name.get(row["author_name"])

        books.append(
            Book(
                goodreads_id=row["goodreads_id"],
                url=row["url"],
                title=row["title"],
                author_id=author.id if author else None,
                summary=row["summary"],
                star_rating=row["star_rating"],
                num_ratings=row["num_ratings"],
                num_reviews=row["num_reviews"],
                genres=row["genres"],
                first_published=row["first_published"],
                kindle_price=row["kindle_price"],
                community_reviews=row["community_reviews"],
            )
        )

    session.add_all(books)

    return {
        "authors_created": authors_created,
        "books_inserted": len(books),
        "books_skipped": len(existing_urls),
    }

def main() -> None:
    read = 0
    books_without_author = 0
    skipped_existing_book = 0
    books_inserted = 0
    authors_created = 0
    df = pl.read_parquet(PARQUET_PATH)
    rows = [transform_row(row) for row in df.iter_rows(named=True)]
    batch_size = 10_000

    with Session(engine) as session:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            result = ingest_batch(session, batch)
            books_inserted += result["books_inserted"]
            authors_created += result["authors_created"]
            skipped_existing_book += result["books_skipped"]
            books_without_author += sum(
                1 for row in batch
                if not row["author_name"]
            )
        session.commit()

    read = len(rows)
    print(f"Read: {read}")
    print(f"Books inserted: {books_inserted}")
    print(f"Authors created: {authors_created}")
    print(f"Books without author: {books_without_author}")
    print(f"Books skipped — already exist: {skipped_existing_book}")


if __name__ == "__main__":
    main()