# Data Discovery

## Dataset

**Source:** BrightData Goodreads Books Dataset

**Origin:** https://huggingface.co/datasets/BrightData/Goodreads-Books

**Records:** 6,389,859 books

**Format:** CSV → Parquet

---

## Available Fields

| Field | Notes |
|--------|------|
| id | Goodreads book identifier |
| url | Goodreads URL |
| name | Book title |
| author | JSON array of author names |
| star_rating | Average Goodreads rating |
| num_ratings | Total number of ratings |
| num_reviews | Number of reviews |
| summary | Book description/summary |
| genres | JSON array of genres |
| first_published | Publication date |
| about_author | JSON object containing author metadata |
| community_reviews | Aggregated review distribution (not review text) |
| kindle_price | Kindle price when available |

---

# Initial Findings

## Dataset Quality

| Observation | Result |
|------------|--------|
| Missing summaries | ~11% |
| Missing authors | Negligible |
| Missing genres (empty strings) | ~73% |
| Community reviews | Rating distribution only |
| Author metadata | Limited metadata (e.g. name, number of books) |

---

## Observations

### Summary

Book summaries are available for the majority of books and appear to be sufficiently descriptive. This is currently the strongest candidate for semantic search and embeddings.

### Genres

Genre coverage is poor. Approximately 73% of records contain empty genre information, making genres unreliable as the primary recommendation signal.

### Authors

Author information is available for nearly all books. The field is stored as a JSON array and will require parsing during ingestion.

### Community Reviews

The dataset does **not** contain textual user reviews.

Instead it contains aggregated statistics such as:

- Number of 1★ reviews
- Number of 2★ reviews
- Percentage distribution

This field is unlikely to be useful for Retrieval-Augmented Generation (RAG).

### Author Metadata

`about_author` currently contains lightweight metadata (for example author name and number of published books). Further inspection is required before deciding whether it should be stored separately.

---

# Candidate Fields for MVP

The following fields appear sufficient for the initial version of BookBuddy.

| Field | Purpose |
|--------|---------|
| id | Primary identifier |
| name | Display |
| author | Display & filtering |
| summary | Semantic search / AI |
| star_rating | Ranking |
| num_ratings | Confidence score |
| first_published | Filtering |
| url | Reference back to Goodreads |

---

# Open Questions

- How many duplicate books exist?
- How long are summaries on average?
- Are summaries available in multiple languages?
- What percentage of books have meaningful (>100 characters) summaries?
- Should books with very few ratings be excluded from recommendations?
- Should author metadata be normalized into a separate table?

---

# Decisions (Current)

- Keep the original CSV as immutable source data.
- Use Parquet for local processing.
- Do not use genres as the primary recommendation signal.
- Do not use `community_reviews` for embeddings or RAG.
- Use book summaries as the primary textual corpus for the MVP.

> This document captures observations from dataset exploration and will evolve as the ingestion pipeline and product requirements become clearer.