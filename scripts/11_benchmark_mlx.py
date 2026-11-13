
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mlx_embedding_models.embedding import EmbeddingModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.database import engine

TEST_SIZE = 2048
BATCH_SIZES = [32, 64, 128]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_test_texts(session: Session) -> list[str]:
    rows = session.execute(
        text("""
            SELECT source_text
            FROM benchmark_book_embeddings
            WHERE embedding_model = :model
            ORDER BY book_id
            LIMIT :limit
        """),
        {
            "model": "BAAI/bge-small-en-v1.5",
            "limit": TEST_SIZE,
        },
    ).scalars().all()

    return list(rows)


def main() -> None:
    with Session(engine) as session:
        texts = fetch_test_texts(session)

    if len(texts) < TEST_SIZE:
        raise ValueError(f"Expected {TEST_SIZE} texts, got {len(texts)}")

    logger.info("Loaded %d texts", len(texts))
    logger.info("Loading BGE-small MLX model")

    model = EmbeddingModel.from_registry("bge-small")

    logger.info("Warming up MLX")

    model.encode(texts[:32])

    results = []

    for batch_size in BATCH_SIZES:
        logger.info("Testing batch size: %d", batch_size)

        start = time.perf_counter()

        for offset in range(0, len(texts), batch_size):
            batch = texts[offset:offset + batch_size]
            model.encode(batch)

        elapsed = time.perf_counter() - start
        throughput = len(texts) / elapsed

        results.append((batch_size, elapsed, throughput))

        logger.info(
            "Batch size %d | Time: %.2f seconds | Throughput: %.2f books/sec",
            batch_size,
            elapsed,
            throughput,
        )

    print("\n========== MLX BENCHMARK RESULTS ==========")

    for batch_size, elapsed, throughput in results:
        print(
            f"Batch size: {batch_size:3d} | "
            f"Time: {elapsed:8.2f}s | "
            f"Throughput: {throughput:8.2f} books/sec"
        )

    best = max(results, key=lambda result: result[2])

    print(
        f"\nBest batch size: {best[0]} "
        f"({best[2]:.2f} books/sec)"
    )


if __name__ == "__main__":
    main()
