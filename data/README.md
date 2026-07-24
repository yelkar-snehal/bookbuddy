# Data

This project uses the Goodreads Books dataset from Hugging Face.

**Dataset:**
https://huggingface.co/datasets/BrightData/Goodreads-Books

## Folder Structure

```
data/
├── raw/
│   └── Goodreads-Books.csv
├── processed/
│   └── Goodreads-Books.parquet
└── embeddings/
```

## Download the Dataset

1. Visit the dataset page:

   https://huggingface.co/datasets/BrightData/Goodreads-Books

2. Download `Goodreads-Books.csv`.

3. Place it in:

   ```
   data/raw/Goodreads-Books.csv
   ```

## Convert to Parquet

Run:

```bash
uv run scripts/01_convert_to_parquet.py
```

This generates:

```
data/processed/Goodreads-Books.parquet
```

## Why Parquet?

Parquet is a compressed, columnar format that is significantly faster to read and more memory-efficient than CSV. It is used throughout the ingestion pipeline.

## Notes

- The dataset is not committed to Git because of its size.
- Download the dataset yourself before running the ingestion scripts.