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
FETCH_SIZE = 2048
ENCODE_BATCH_SIZE = 128
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


def fetch_next_books_without_embeddings(
    session: Session,
    after_id: int,
    limit: int,
) -> list[dict]:
    # Keyset pagination: start after the last book we saw, so each batch
    # reads from the primary key index instead of re-scanning every book
    # that is already embedded.
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
            WHERE books.id > :after_id
                AND NOT EXISTS (
                    SELECT 1
                    FROM book_embeddings
                    WHERE book_embeddings.book_id = books.id
                )
            ORDER BY books.id
            LIMIT :limit
            """
        ),
        {"after_id": after_id, "limit": limit},
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
    # On Apple Silicon this should be "mps". "cpu" means the GPU is unused.
    logger.info("Embedding device: %s", model.device)

    with Session(engine) as session:
        processed = 0
        # The first fetch skips every embedded book once. After that, each
        # fetch starts from the last book id, so the cost per batch stays flat.
        last_book_id = 0

        while BOOK_LIMIT is None or processed < BOOK_LIMIT:
            remaining = BOOK_LIMIT - processed if BOOK_LIMIT is not None else FETCH_SIZE
            batch_limit = min(FETCH_SIZE, remaining)
            batch_start = time.perf_counter()
            batch = fetch_next_books_without_embeddings(
                session,
                last_book_id,
                batch_limit,
            )

            if not batch:
                logger.info("No more books without embeddings")
                break

            last_book_id = batch[-1]["id"]
            fetch_seconds = time.perf_counter() - batch_start

            for book in batch:
                book["source_text"] = build_source_text(book)

            embeddings = model.encode(
                [book["source_text"] for book in batch],
                batch_size=ENCODE_BATCH_SIZE,
                normalize_embeddings=True,
            ).tolist()

            insert_embeddings(session, batch, embeddings)
            session.commit()

            processed += len(batch)
            batch_seconds = time.perf_counter() - batch_start
            elapsed = time.perf_counter() - start_time
            # batch_rate shows the current speed. The overall rate is an
            # average since start and hides a slowdown.
            logger.info(
                "Embedded %d books | batch_rate=%.2f books/sec | "
                "fetch=%.2fs | overall_rate=%.2f books/sec | last_book_id=%d",
                processed,
                len(batch) / batch_seconds if batch_seconds else 0,
                fetch_seconds,
                processed / elapsed if elapsed else 0,
                last_book_id,
            )


if __name__ == "__main__":
    main()
