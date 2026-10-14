import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.database import engine


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
BATCH_SIZE = 128
BOOK_LIMIT = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def build_source_text(book: dict) -> str:
    parts = [
        f"Title: {book['title']}",
    ]

    if book["author_name"]:
        parts.append(f"Author: {book['author_name']}")

    if book["genres"]:
        parts.append(f"Genres: {book['genres']}")

    if book["summary"]:
        parts.append(f"Summary: {book['summary']}")

    return "\n".join(parts)


def vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(value) for value in embedding) + "]"


def fetch_next_books_without_embeddings(session: Session, limit: int) -> list[dict]:
    rows = session.execute(
        text(
            """
            SELECT
                books.id,
                books.title,
                authors.name AS author_name,
                books.genres,
                books.summary
            FROM books
            LEFT JOIN authors ON authors.id = books.author_id
            LEFT JOIN book_embeddings ON book_embeddings.book_id = books.id
            WHERE book_embeddings.id IS NULL
            ORDER BY books.id
            LIMIT :limit
            """
        ),
        {"limit": limit},
    ).mappings().all()

    return [dict(row) for row in rows]


def insert_embeddings(
    session: Session,
    rows: list[dict],
    embeddings: list[list[float]],
) -> None:
    session.execute(
        text(
            """
            INSERT INTO book_embeddings (
                book_id,
                embedding_model,
                embedding_dimension,
                source_text,
                embedding
            )
            VALUES (
                :book_id,
                :embedding_model,
                :embedding_dimension,
                :source_text,
                CAST(:embedding AS vector)
            )
            ON CONFLICT (book_id) DO NOTHING
            """
        ),
        [
            {
                "book_id": row["id"],
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "source_text": row["source_text"],
                "embedding": vector_literal(embedding),
            }
            for row, embedding in zip(rows, embeddings)
        ],
    )


def main() -> None:
    start_time = time.perf_counter()
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    with Session(engine) as session:
        processed = 0

        while BOOK_LIMIT is None or processed < BOOK_LIMIT:
            remaining = BOOK_LIMIT - processed if BOOK_LIMIT is not None else BATCH_SIZE
            batch_limit = min(BATCH_SIZE, remaining)
            batch = fetch_next_books_without_embeddings(session, batch_limit)

            if not batch:
                logger.info("No more books without embeddings")
                break

            for book in batch:
                book["source_text"] = build_source_text(book)

            embeddings = model.encode(
                [book["source_text"] for book in batch],
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
            ).tolist()

            insert_embeddings(session, batch, embeddings)
            session.commit()

            processed += len(batch)
            elapsed = time.perf_counter() - start_time
            rate = processed / elapsed if elapsed else 0
            logger.info(
                "Embedded %d books | rate=%.2f books/sec",
                processed,
                rate,
            )


if __name__ == "__main__":
    main()
