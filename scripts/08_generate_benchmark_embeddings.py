
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.database import engine


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
FETCH_SIZE = 256
ENCODE_BATCH_SIZE = 32
SAMPLE_SIZE = 10_000
RANDOM_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def build_source_text(book: dict) -> str:
    parts = [f"Title: {book['title']}"]

    if book["author_name"]:
        parts.append(f"Author: {book['author_name']}")

    if book["genres"]:
        parts.append(f"Genres: {book['genres']}")

    if book["summary"]:
        parts.append(f"Summary: {book['summary']}")

    return "\n".join(parts)


def vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(value) for value in embedding) + "]"


def load_expected_books(session: Session) -> list[dict]:
    root = Path(__file__).resolve().parents[1]
    queries_path = root / "eval" / "retrieval_queries.json"
    queries = json.loads(queries_path.read_text())["queries"]

    expected_books = {}

    for query in queries:
        for expected in query["expected"]:
            rows = session.execute(
                text("""
                    SELECT books.id, books.title
                    FROM books
                    JOIN authors ON authors.id = books.author_id
                    WHERE regexp_replace(books.title, '\\s+', ' ', 'g')
                          ILIKE :title_pattern
                      AND regexp_replace(authors.name, '\\s+', ' ', 'g')
                          ILIKE :author_pattern
                """),
                {
                    "title_pattern": expected["title"] + "%",
                    "author_pattern": "%" + expected["author"] + "%",
                },
            ).mappings().all()

            for row in rows:
                expected_books[row["id"]] = dict(row)

    return list(expected_books.values())


def fetch_books_by_ids(
    session: Session,
    book_ids: list[int],
) -> list[dict]:
    if not book_ids:
        return []

    rows = session.execute(
        text("""
            SELECT
                books.id,
                books.title,
                authors.name AS author_name,
                books.genres,
                books.summary
            FROM books
            LEFT JOIN authors ON authors.id = books.author_id
            WHERE books.id = ANY(:book_ids)
              AND NOT EXISTS (
                  SELECT 1
                  FROM benchmark_book_embeddings
                  WHERE benchmark_book_embeddings.book_id = books.id
                    AND benchmark_book_embeddings.embedding_model = :model
              )
            ORDER BY books.id
        """),
        {
            "book_ids": book_ids,
            "model": EMBEDDING_MODEL,
        },
    ).mappings().all()

    return [dict(row) for row in rows]


def fetch_benchmark_books(
    session: Session,
    offset: int,
    limit: int,
) -> list[dict]:
    rows = session.execute(
        text("""
            SELECT
                books.id,
                books.title,
                authors.name AS author_name,
                books.genres,
                books.summary
            FROM books
            LEFT JOIN authors ON authors.id = books.author_id
            WHERE NOT EXISTS (
                SELECT 1
                FROM benchmark_book_embeddings
                WHERE benchmark_book_embeddings.book_id = books.id
                  AND benchmark_book_embeddings.embedding_model = :model
            )
            ORDER BY md5(books.id::text || :seed)
            OFFSET :offset
            LIMIT :limit
        """),
        {
            "model": EMBEDDING_MODEL,
            "seed": str(RANDOM_SEED),
            "offset": offset,
            "limit": limit,
        },
    ).mappings().all()

    return [dict(row) for row in rows]


def insert_benchmark_embeddings(
    session: Session,
    rows: list[dict],
    embeddings: list[list[float]],
    embedding_dimension: int,
) -> None:
    session.execute(
        text("""
            INSERT INTO benchmark_book_embeddings (
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
            ON CONFLICT (book_id, embedding_model) DO NOTHING
        """),
        [
            {
                "book_id": row["id"],
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimension": embedding_dimension,
                "source_text": row["source_text"],
                "embedding": vector_literal(embedding),
            }
            for row, embedding in zip(rows, embeddings)
        ],
    )


def generate_embeddings(
    session: Session,
    model: SentenceTransformer,
    books: list[dict],
    embedding_dimension: int,
) -> int:
    if not books:
        return 0

    processed = 0

    for start in range(0, len(books), FETCH_SIZE):
        batch = books[start : start + FETCH_SIZE]

        for book in batch:
            book["source_text"] = build_source_text(book)

        embeddings = model.encode_document(
            [book["source_text"] for book in batch],
            batch_size=ENCODE_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        insert_benchmark_embeddings(
            session,
            batch,
            embeddings,
            embedding_dimension,
        )
        session.commit()

        processed += len(batch)

        logger.info(
            "Generated %d / %d embeddings in this phase",
            processed,
            len(books),
        )

    return processed


def main() -> None:
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)

    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dimension = model.get_embedding_dimension()

    logger.info("Model: %s", EMBEDDING_MODEL)
    logger.info("Dimensions: %d", embedding_dimension)
    logger.info("Max sequence length: %d", model.max_seq_length)
    logger.info("Device: %s", model.device)

    with Session(engine) as session:
        expected_books = load_expected_books(session)
        expected_book_ids = [book["id"] for book in expected_books]

        expected_batch = fetch_books_by_ids(
            session,
            expected_book_ids,
        )

        logger.info(
            "Expected editions: %d resolved, %d need embeddings",
            len(expected_books),
            len(expected_batch),
        )

        expected_processed = generate_embeddings(
            session,
            model,
            expected_batch,
            embedding_dimension,
        )

        logger.info(
            "Expected-edition phase complete: %d new embeddings",
            expected_processed,
        )

        remaining_sample_size = SAMPLE_SIZE
        sample_processed = 0
        offset = 0

        while sample_processed < remaining_sample_size:
            batch_limit = min(
                FETCH_SIZE,
                remaining_sample_size - sample_processed,
            )

            batch = fetch_benchmark_books(
                session,
                offset,
                batch_limit,
            )

            if not batch:
                logger.info("No more eligible catalogue books")
                break

            processed = generate_embeddings(
                session,
                model,
                batch,
                embedding_dimension,
            )

            sample_processed += processed
            offset += len(batch)

            logger.info(
                "Random sample progress: %d / %d",
                sample_processed,
                remaining_sample_size,
            )

        total = session.execute(
            text("""
                SELECT COUNT(*)
                FROM benchmark_book_embeddings
                WHERE embedding_model = :model
            """),
            {"model": EMBEDDING_MODEL},
        ).scalar_one()

        logger.info("Total BGE-small embeddings in database: %d", total)


if __name__ == "__main__":
    main()
