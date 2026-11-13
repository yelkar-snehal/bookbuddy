
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.database import engine


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
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
            "model": EMBEDDING_MODEL,
            "limit": TEST_SIZE,
        },
    ).scalars().all()

    return list(rows)


def synchronize_device(device: str) -> None:
    if device.startswith("mps"):
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def benchmark_batch_size(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> float:
    synchronize_device(str(model.device))

    start = time.perf_counter()

    model.encode_document(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    synchronize_device(str(model.device))

    elapsed = time.perf_counter() - start
    return elapsed


def main() -> None:
    with Session(engine) as session:
        texts = fetch_test_texts(session)

    if len(texts) < TEST_SIZE:
        raise ValueError(
            f"Expected {TEST_SIZE} texts, found {len(texts)}"
        )

    logger.info("Loaded %d existing book texts", len(texts))

    logger.info("Loading model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    logger.info("Device: %s", model.device)
    logger.info("Max sequence length: %d", model.max_seq_length)

    # Warm up the model before measuring.
    logger.info("Warming up model")

    model.encode_document(
        texts[:32],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    synchronize_device(str(model.device))

    results = []

    for batch_size in BATCH_SIZES:
        logger.info("Testing batch size: %d", batch_size)

        try:
            elapsed = benchmark_batch_size(
                model,
                texts,
                batch_size,
            )

            throughput = len(texts) / elapsed

            results.append({
                "batch_size": batch_size,
                "elapsed": elapsed,
                "throughput": throughput,
            })

            logger.info(
                "Batch size %d | Time: %.2f seconds | "
                "Throughput: %.2f books/sec",
                batch_size,
                elapsed,
                throughput,
            )

        except RuntimeError:
            logger.exception(
                "Batch size %d failed, possibly due to memory limits",
                batch_size,
            )

            if str(model.device).startswith("mps"):
                torch.mps.empty_cache()

    print("\n========== BENCHMARK RESULTS ==========")

    for result in results:
        print(
            f"Batch size: {result['batch_size']:3d} | "
            f"Time: {result['elapsed']:8.2f}s | "
            f"Throughput: {result['throughput']:8.2f} books/sec"
        )

    if results:
        best = max(results, key=lambda item: item["throughput"])

        print(
            f"\nBest batch size: {best['batch_size']} "
            f"({best['throughput']:.2f} books/sec)"
        )


if __name__ == "__main__":
    main()
