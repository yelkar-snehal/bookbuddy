# BookBuddy

An AI-powered book discovery platform focused on personalized, explainable recommendations, built on 6.4 million Goodreads books.

I am building BookBuddy to learn AI engineering step by step: data engineering, semantic search, retrieval evaluation, RAG, agents and MCP. Each feature starts from a real problem in the project and has a measured result. [docs/learning-log.md](docs/learning-log.md) records each problem, the fix, and the numbers before and after.

## Current state

| Area | Status |
|---|---|
| Data pipeline | 6,389,859 books and 2,112,054 authors from the BrightData Goodreads dataset, profiled, cleaned and loaded into PostgreSQL |
| REST API | FastAPI endpoints for books and authors, with title and author filters |
| Semantic search | 384-dimension embeddings (`all-MiniLM-L6-v2`) for every book, stored in pgvector, cosine distance search |
| Retrieval evaluation | 26 test queries in 5 types, scored with recall@10, MRR and hit rate |
| RAG recommendations | Next milestone |

## Results so far

### Embedding generation

The first version of the embedding script got slower as it ran, because each batch rescanned every book that was already embedded. Keyset pagination fixed it.

| | Before | After |
|---|---|---|
| Speed | about 44 books/s | 350 to 370 books/s |

### Search quality

Baseline semantic search found the expected book in the top 10 for 11.5% of the test queries. Study guides and summaries ranked above the real books, because they repeat the title and plot of the original. A filter on the number of ratings removes them:

| min_ratings | Share of catalogue searched | recall@10 | MRR |
|---|---|---|---|
| none | 100% | 0.115 | 0.063 |
| 100 | 9.2% | 0.231 | 0.125 |
| 1,000 | 2.1% | 0.327 | 0.232 |
| 10,000 | 0.3% | 0.404 | 0.273 |

The filter improves recall but hides most of the catalogue. The next experiment uses popularity in the ranking instead of as a filter. The [learning log](docs/learning-log.md) has the full analysis.

## Tech stack

- Python 3.12, managed with uv
- PostgreSQL 13 with pgvector, in Docker
- SQLAlchemy and Alembic
- FastAPI
- Polars and DuckDB for data processing
- sentence-transformers for embeddings

## Run it locally

You need Docker and [uv](https://docs.astral.sh/uv/). The full dataset needs a lot of disk space: the embedding vectors alone are about 10 GB.

1. Start the database:

   ```bash
   docker compose up -d
   ```

2. Create `.env` from `.env.sample`. For the Docker database, use:

   ```
   DATABASE_URL=postgresql+psycopg://bookbuddy_user:bookbuddy_password@localhost:5432/bookbuddy_db
   ```

3. Install the dependencies and create the tables:

   ```bash
   uv sync
   uv run alembic upgrade head
   ```

4. Download the dataset to `data/raw/Goodreads-Books.csv`. [data/README.md](data/README.md) gives the steps.

5. Convert the dataset, load it, and create the embeddings:

   ```bash
   uv run scripts/01_convert_dataset.py
   uv run scripts/05_ingest_sample.py
   uv run scripts/06_generate_book_embeddings.py
   ```

   `05_ingest_sample.py` loads all books. At about 360 books/s, embedding all 6.4 million books takes about 5 hours on an Apple Silicon laptop.

6. Start the API:

   ```bash
   uv run uvicorn app.main:app --reload
   ```

   Open http://localhost:8000/docs for the interactive API documentation. Example search:

   ```
   GET /search/books?q=a ship captain obsessed with a white whale&min_ratings=1000
   ```

7. Run the retrieval evaluation:

   ```bash
   uv run scripts/07_evaluate_retrieval.py --min-ratings 1000
   ```

## Project structure

```
app/            FastAPI application: API routes, database models, schemas
scripts/        Data processing, ingestion, embedding and evaluation scripts
eval/           Evaluation queries and saved results
migrations/     Alembic database migrations
docs/           Architecture, data discovery, database design, learning log
data/           Dataset instructions (the data itself is not in Git)
```

## Roadmap

1. Data and search foundation: done
2. Retrieval evaluation and search improvements: in progress
3. RAG: grounded book recommendations with references to the retrieved books
4. Agent with tool calling (LangGraph), single agent first
5. MCP server for BookBuddy capabilities
6. Evaluation and observability for answers, latency and model calls
7. Production engineering: tests, CI/CD, security, deployment

[docs/architecture.md](docs/architecture.md) describes each phase.
