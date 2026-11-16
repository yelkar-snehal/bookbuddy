
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lingua import Language, LanguageDetectorBuilder
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from app.db.database import engine


MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BGE_MODEL = "BAAI/bge-small-en-v1.5"

BATCH_SIZE = 512


def load_candidate_texts():
    query = text("""
        SELECT
            bge.book_id,
            bge.source_text AS bge_source_text,
            mini.source_text AS minilm_source_text
        FROM benchmark_book_embeddings bge
        JOIN book_embeddings mini
            ON mini.book_id = bge.book_id
        WHERE bge.embedding_model = :bge_model
          AND mini.embedding_model = :minilm_model
        ORDER BY bge.book_id
    """)

    with engine.connect() as connection:
        rows = connection.execute(
            query,
            {
                "bge_model": BGE_MODEL,
                "minilm_model": MINILM_MODEL,
            },
        ).mappings().all()

    print(f"Books with both embeddings: {len(rows):,}")

    return rows


def filter_english_candidates(rows):
    detector = LanguageDetectorBuilder.from_all_languages().build()

    english_rows = []

    for index, row in enumerate(rows, start=1):
        language = detector.detect_language_of(
            row["bge_source_text"]
        )

        if language == Language.ENGLISH:
            english_rows.append(row)

        if index % 2000 == 0:
            print(
                f"Language detection: {index:,}/{len(rows):,}",
                flush=True,
            )

    print(f"English candidate pool: {len(english_rows):,}")

    return english_rows


def measure_truncation(
    rows,
    source_column,
    model_name,
):
    print(f"\nLoading tokenizer: {model_name}")

    model = SentenceTransformer(model_name)
    tokenizer = model.tokenizer
    max_length = model.max_seq_length

    total = 0
    truncated = 0
    max_tokens = 0

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]

        texts = [row[source_column] or "" for row in batch]

        tokenized = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            verbose=False,
        )

        for tokens in tokenized["input_ids"]:
            token_count = len(tokens)

            total += 1
            max_tokens = max(max_tokens, token_count)

            if token_count > max_length:
                truncated += 1

        if total % 2000 == 0 or total == len(rows):
            print(
                f"{model_name}: {total:,}/{len(rows):,} | "
                f"truncated={truncated:,}",
                flush=True,
            )

    percentage = (
        truncated / total * 100
        if total else 0
    )

    print(f"\nResults: {model_name}")
    print(f"Records checked: {total:,}")
    print(f"Max sequence length: {max_length}")
    print(f"Truncated records: {truncated:,}")
    print(f"Truncation rate: {percentage:.2f}%")
    print(f"Longest source: {max_tokens:,} tokens")

    return {
        "model": model_name,
        "total": total,
        "truncated": truncated,
        "percentage": percentage,
        "max_length": max_length,
    }


def main():
    rows = load_candidate_texts()
    english_rows = filter_english_candidates(rows)

    minilm_results = measure_truncation(
        english_rows,
        "minilm_source_text",
        MINILM_MODEL,
    )

    bge_results = measure_truncation(
        english_rows,
        "bge_source_text",
        BGE_MODEL,
    )

    print("\n========== COMPARISON ==========")

    for result in [minilm_results, bge_results]:
        print(
            f"{result['model']}: "
            f"{result['truncated']:,}/{result['total']:,} "
            f"({result['percentage']:.2f}%) "
            f"| limit={result['max_length']}"
        )


if __name__ == "__main__":
    main()
