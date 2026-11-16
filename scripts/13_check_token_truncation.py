
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from app.db.database import engine

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 2000

model = SentenceTransformer(MODEL_NAME)
tokenizer = model.tokenizer
max_length = model.max_seq_length

query = text("""
    SELECT source_text
    FROM book_embeddings
    WHERE embedding_model = :model
    ORDER BY book_id
""")

total = 0
truncated = 0

with engine.connect() as connection:
    result = connection.execution_options(
        stream_results=True,
        yield_per=BATCH_SIZE,
    ).execute(query, {"model": MODEL_NAME})

    while True:
        rows = result.fetchmany(BATCH_SIZE)

        if not rows:
            break

        texts = [row.source_text for row in rows]

        tokenized = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            verbose=False,
        )

        for tokens in tokenized["input_ids"]:
            total += 1
            if len(tokens) > max_length:
                truncated += 1

        if total % 10000 == 0:
            print(
                f"Processed: {total:,} | "
                f"Truncated: {truncated:,} | "
                f"Rate: {truncated / total * 100:.2f}%",
                flush=True,
            )

print("\n--- Results ---")
print(f"Total records: {total:,}")
print(f"Truncated records: {truncated:,}")
print(f"Truncation rate: {truncated / total * 100:.2f}%")
