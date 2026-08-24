from decimal import Decimal
import json

import polars as pl
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import engine
from app.db.models import Author, Book


PARQUET_PATH = "data/processed/Goodreads-Books.parquet"
SAMPLE_SIZE = 10_000


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

def ingest_row(session: Session, data: dict) -> tuple[bool, bool, bool]:
    author = None

    if data["author_name"]:
        author = session.scalar(
            select(Author).where(Author.name == data["author_name"])
        )

        if author is None:
            author = Author(
                name=data["author_name"],
                num_books=data["author_metadata"]["num_books"],
                num_followers=data["author_metadata"]["num_followers"],
                about=data["author_metadata"]["about"],
            )
            session.add(author)
            session.flush()

            author_created = True
        else:
            author_created = False
    else:
        author_created = False

    book = session.scalar(
        select(Book).where(Book.url == data["url"])
    )

    if book is not None:
        return False, author_created, True

    book = Book(
        goodreads_id=data["goodreads_id"],
        url=data["url"],
        title=data["title"],
        author_id=author.id if author else None,
        summary=data["summary"],
        star_rating=data["star_rating"],
        num_ratings=data["num_ratings"],
        num_reviews=data["num_reviews"],
        genres=data["genres"],
        first_published=data["first_published"],
        kindle_price=data["kindle_price"],
        community_reviews=data["community_reviews"],
    )

    session.add(book)

    return True, author_created, False

def main() -> None:
    read = 0
    books_without_author = 0
    skipped_existing_book = 0
    books_inserted = 0
    authors_created = 0
    df = pl.read_parquet(PARQUET_PATH).head(SAMPLE_SIZE)

    with Session(engine) as session:
        for row in df.iter_rows(named=True):
            read += 1
            data = transform_row(row)
            book_inserted, author_created, book_skipped = ingest_row(
                session, data
            )
            if not data["author_name"]:
                books_without_author += 1

            if book_inserted:
                books_inserted += 1

            if author_created:
                authors_created += 1

            if book_skipped:
                skipped_existing_book += 1
        session.commit()

    print(f"Read: {read}")
    print(f"Books inserted: {books_inserted}")
    print(f"Authors created: {authors_created}")
    print(f"Books without author: {books_without_author}")
    print(f"Books skipped — already exist: {skipped_existing_book}")


if __name__ == "__main__":
    main()