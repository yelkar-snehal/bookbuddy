
import logging
import random
import sys
from pathlib import Path

import numpy as np
from lingua import Language, LanguageDetectorBuilder
from sentence_transformers import SentenceTransformer
from mlx_embedding_models.embedding import EmbeddingModel
from sqlalchemy import text
from sqlalchemy.orm import Session

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.db.database import engine


MODEL_NAME = "BAAI/bge-small-en-v1.5"
TEST_SIZE = 128
RANDOM_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_test_data(session: Session) -> list[dict]:
    rows = session.execute(
        text("""
            SELECT
                book_id,
                source_text,
                embedding::text AS embedding
            FROM benchmark_book_embeddings
            WHERE embedding_model = :model
            ORDER BY book_id
        """),
        {"model": MODEL_NAME},
    ).mappings().all()

    return [dict(row) for row in rows]


def select_english_books(rows: list[dict]) -> list[dict]:
    logger.info("Initializing language detector")

    detector = LanguageDetectorBuilder.from_all_languages().build()

    english_books = []

    for row in rows:
        detected_language = detector.detect_language_of(
            row["source_text"]
        )

        if detected_language == Language.ENGLISH:
            english_books.append(row)

    logger.info(
        "Detected %d English books out of %d records",
        len(english_books),
        len(rows),
    )

    if len(english_books) < TEST_SIZE:
        raise ValueError(
            f"Need {TEST_SIZE} English books, "
            f"found {len(english_books)}"
        )

    rng = random.Random(RANDOM_SEED)

    return rng.sample(english_books, TEST_SIZE)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    return vectors / np.maximum(norms, 1e-12)


def compare(
    name: str,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    similarities = np.sum(first * second, axis=1)

    print(f"\n{name}")
    print(f"Mean cosine similarity: {similarities.mean():.6f}")
    print(f"Median: {np.median(similarities):.6f}")
    print(f"5th percentile: {np.percentile(similarities, 5):.6f}")
    print(f"Minimum: {similarities.min():.6f}")
    print(f"Below 0.99: {(similarities < 0.99).sum()}")
    print(f"Below 0.95: {(similarities < 0.95).sum()}")

    return similarities


def main() -> None:
    with Session(engine) as session:
        rows = fetch_test_data(session)

    logger.info("Loaded %d benchmark records", len(rows))

    selected_books = select_english_books(rows)

    texts = [
        book["source_text"]
        for book in selected_books
    ]

    stored_embeddings = np.array([
        np.fromstring(
            book["embedding"].strip("[]"),
            sep=",",
        )
        for book in selected_books
    ])

    logger.info("Selected %d English books", len(texts))

    logger.info("Loading PyTorch BGE-small")

    torch_model = SentenceTransformer(MODEL_NAME)

    logger.info("Loading MLX BGE-small")

    mlx_model = EmbeddingModel.from_registry("bge-small")

    logger.info("Generating PyTorch embeddings")

    torch_embeddings = torch_model.encode_document(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    logger.info("Generating MLX embeddings")

    mlx_embeddings = mlx_model.encode(texts)

    stored_embeddings = normalize(stored_embeddings)
    torch_embeddings = normalize(np.asarray(torch_embeddings))
    mlx_embeddings = normalize(np.asarray(mlx_embeddings))

    print("\n========== ENGLISH-ONLY MLX VALIDATION ==========")

    print(f"Stored shape: {stored_embeddings.shape}")
    print(f"PyTorch shape: {torch_embeddings.shape}")
    print(f"MLX shape: {mlx_embeddings.shape}")

    stored_torch = compare(
        "STORED vs PYTORCH",
        stored_embeddings,
        torch_embeddings,
    )

    stored_mlx = compare(
        "STORED vs MLX",
        stored_embeddings,
        mlx_embeddings,
    )

    torch_mlx = compare(
        "PYTORCH vs MLX",
        torch_embeddings,
        mlx_embeddings,
    )

    print("\n========== WORST MLX MATCHES ==========")

    worst_indices = np.argsort(stored_mlx)[:10]

    for index in worst_indices:
        print(
            f"Book ID: {selected_books[index]['book_id']} | "
            f"Stored/MLX: {stored_mlx[index]:.6f} | "
            f"Stored/PyTorch: {stored_torch[index]:.6f} | "
            f"PyTorch/MLX: {torch_mlx[index]:.6f}"
        )


if __name__ == "__main__":
    main()
