
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.database import engine


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TEST_SIZE = 2048
BATCH_SIZES = [32, 64, 128]

ROOT = Path(__file__).resolve().parents[1]
ONNX_MODEL_DIR = ROOT / "models" / "bge-small-en-onnx"

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


def load_onnx_model() -> SentenceTransformer:
    if ONNX_MODEL_DIR.exists():
        logger.info("Loading existing ONNX model from %s", ONNX_MODEL_DIR)

        return SentenceTransformer(
            str(ONNX_MODEL_DIR),
            backend="onnx",
            model_kwargs={
                "provider": "CPUExecutionProvider",
            },
        )

    logger.info("Exporting BGE-small to ONNX for the first time")

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        backend="onnx",
        model_kwargs={
            "provider": "CPUExecutionProvider",
        },
    )

    ONNX_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ONNX_MODEL_DIR))

    logger.info("ONNX model saved to %s", ONNX_MODEL_DIR)

    return model


def benchmark_batch_size(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> float:
    start = time.perf_counter()

    model.encode_document(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

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

    model = load_onnx_model()

    logger.info("Model: %s", EMBEDDING_MODEL)
    logger.info("Backend: ONNX Runtime")
    logger.info("Device: CPUExecutionProvider")
    logger.info("Max sequence length: %d", model.max_seq_length)

    # Warm up the runtime.
    logger.info("Warming up ONNX Runtime")

    model.encode_document(
        texts[:32],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

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

        except Exception:
            logger.exception(
                "Batch size %d failed",
                batch_size,
            )

    print("\n========== ONNX BENCHMARK RESULTS ==========")

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
