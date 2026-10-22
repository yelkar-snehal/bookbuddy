# Learning Log

This log records what I built in BookBuddy, what I measured, and what I learned. Each entry starts from a real problem in the project. Where a change could make things better or worse, the entry gives the numbers before and after.

---

## 1. Embedding generation: from 44 to about 360 books per second

Date: 2026-10-16
Code: `scripts/06_generate_book_embeddings.py` (commit `19c696b`)

### Problem

The script creates one embedding for each of the 6,389,859 books with `sentence-transformers/all-MiniLM-L6-v2` and stores it in `book_embeddings`. After more than 30 hours it had embedded 4,790,144 books, an average of about 44 books per second. The laptop ran at full load for the whole time.

The log showed only the average rate since the start. That number hid the real problem.

### Cause

Each batch selected the next books without an embedding with this query:

```sql
SELECT ...
FROM books
LEFT JOIN book_embeddings ON book_embeddings.book_id = books.id
WHERE book_embeddings.id IS NULL
ORDER BY books.id
LIMIT 128
```

The script embeds books in id order, so the books that are done have the lowest ids. To find the first book without an embedding, PostgreSQL must pass over every book that is already embedded. With 4.79 million books done, each batch of 128 books first read millions of rows. The cost of each batch increased with progress, so the total time increased approximately with the square of the number of books.

### Fix

I changed the query to keyset pagination. The script keeps the last book id that it processed and starts the next batch after it:

```sql
WHERE books.id > :last_book_id
  AND NOT EXISTS (
      SELECT 1 FROM book_embeddings
      WHERE book_embeddings.book_id = books.id
  )
ORDER BY books.id
LIMIT 2048
```

`books.id` is the primary key, so PostgreSQL goes directly to `last_book_id` in the index and reads only the next rows. `NOT EXISTS` uses the unique index on `book_embeddings.book_id`.

I also made these changes:

- The script fetches 2,048 books for each database round trip. The model encodes them in chunks of 128.
- The log shows the rate of the current batch and the fetch time, not only the average.
- The log shows the device of the model at startup, to confirm that the Apple GPU (`mps`) is in use.

The script is idempotent. `NOT EXISTS` skips books that have an embedding, and `ON CONFLICT (book_id) DO NOTHING` ignores duplicates. Each batch is one transaction, so I can stop and restart the script at any time.

### Result

| | Before | After |
|---|---|---|
| Rate | about 44 books/s (average over 30 hours) | 350 to 370 books/s (steady) |
| Fetch time for each batch | increased with progress | 0.22 s for 2,048 books |
| Bottleneck | database | model on the GPU |

On a test database with 2 million books and 1.5 million embeddings, the old query took 742 ms for 128 books (5.8 ms for each book). The new query took 124 ms for 2,048 books (0.06 ms for each book).

### What I learned

- An average rate hides a slowdown. Log the rate of the current batch.
- `OFFSET n` and "find the first row that is not done" both get slower as `n` increases. `WHERE id > last_id` with an index has the same cost at every position.
- I did not change the model settings (precision, maximum sequence length) during the run. A change would give different vectors for the books embedded before and after the change, and the evaluation would be less reliable.

---

## 2. Retrieval evaluation: a baseline for semantic search

Date: 2026-10-19
Code: `eval/retrieval_queries.json`, `scripts/07_evaluate_retrieval.py` (commit `b914466`)

### Problem

The semantic search endpoint (`GET /search/books`) returned results, but I had no way to say if the results were good. Without a number, I cannot show that a change makes search better.

### What I built

An evaluation set of 26 queries. Each query has one or more expected books. The queries have five types, so that the results show where search is strong and where it is weak:

| Type | Queries | Example |
|---|---|---|
| `plot` | 11 | "a ship captain is obsessed with hunting a white whale" |
| `mood` | 5 | "dark academia about a group of classics students who commit a murder" |
| `similar_to` | 4 | "books like Gone Girl" |
| `lookup` | 4 | "Dune by Frank Herbert" |
| `nonfiction` | 2 | "how habits form and how to change them" |

The evaluation script calls the API function directly, so it always measures the same SQL and model as the endpoint. It reports three metrics:

- recall@10: the share of expected books that are in the top 10 results.
- MRR (mean reciprocal rank): 1 divided by the rank of the first expected book, averaged over the queries. A value of 1.0 means that the first result is always correct.
- hit rate: the share of queries with at least one expected book in the top 10.

The script matches an expected book by title prefix and author name, not by id. Goodreads has many editions of one book, and each edition is a separate row. Any edition counts as a hit, and each expected book counts only once.

### Data problems that the evaluation found

The first run could not find 8 expected books. The causes were problems in the dataset:

- Some author names have repeated spaces, for example `Becky  Chambers`. The script now collapses whitespace before it compares names. Because `authors.name` is unique, `Becky Chambers` and `Becky  Chambers` can be two different authors. The ingestion pipeline must clean this.
- Some books are stored under a foreign edition. *Gone Girl* is stored as `L'amore bugiardo - La versione di Nick` (Italian), with 3.1 million ratings. MiniLM is an English model, so a foreign-language summary gives a poor embedding.
- Some well-known books are not in the dataset: *The Martian*, *Project Hail Mary*, *Legends & Lattes* and *The Silent Patient*. I replaced these queries with books that are in the dataset.

### Baseline result

| Metric | Value |
|---|---|
| recall@10 | 0.115 |
| MRR | 0.063 |
| hit rate | 0.115 (3 of 26 queries) |

The main failure was clear in the top results. Study guides and summaries, for example `The Martian Chronicles (SparkNotes)`, `The Hunger Games Discussion Guide` and `Notes on Pride and Prejudice`, ranked above the books themselves. These guides repeat the title and plot of the original book, so their embeddings are very close to the query. Obscure books with similar words filled the remaining places. Vector similarity measures closeness in meaning only. It has no signal for how well known or how good a book is.

### What I learned

- Build the evaluation before you improve search. The first run found data problems that I did not know about.
- The evaluation set must match the dataset. Before I trust a score, I must check that every expected book exists.

---

## 3. Popularity filter: recall against coverage

Date: 2026-10-19
Code: `app/api/search.py` (optional `min_ratings` parameter), `scripts/07_evaluate_retrieval.py --min-ratings`

### Experiment

I added an optional `min_ratings` parameter to `GET /search/books`. When it is set, search returns only books with at least that many ratings. When it is not set, search behaves as before. I ran the evaluation with no filter and with three thresholds.

### Result

| min_ratings | Books searched | Share of catalogue | recall@10 | MRR | hit rate |
|---|---|---|---|---|---|
| none | 6,389,859 | 100% | 0.115 | 0.063 | 0.115 |
| 100 | 585,076 | 9.2% | 0.231 | 0.125 | 0.269 |
| 1,000 | 132,108 | 2.1% | 0.327 | 0.232 | 0.385 |
| 10,000 | 19,775 | 0.3% | 0.404 | 0.273 | 0.462 |

The run with no filter gave the same numbers as the baseline. This confirmed that the change did not alter the default behaviour.

Results by query type:

| min_ratings | lookup | mood | nonfiction | plot | similar_to |
|---|---|---|---|---|---|
| none | 0.25 | 0.0 | 0.0 | 0.182 | 0.0 |
| 100 | 1.0 | 0.0 | 0.5 | 0.091 | 0.0 |
| 1,000 | 1.0 | 0.2 | 0.5 | 0.227 | 0.0 |
| 10,000 | 0.75 | 0.6 | 0.75 | 0.227 | 0.125 |

(Values are recall@10.)

### Analysis

- 91% of the books have fewer than 100 ratings. This long tail of obscure books and study guides buried the books that readers look for.
- The highest score comes from searching only 0.3% of the catalogue. A reader who looks for a new or niche book would never find it with this filter.
- No threshold is the best for every query type. At 1,000 ratings, lookup has its best MRR (0.675). At 10,000 ratings, mood has its best recall (0.6). This shows that a hard cut-off is the wrong tool.
- For some plot queries, recall decreased at 100 ratings. The edition that matched had fewer than 100 ratings and the filter removed it, while the popular edition was not in the top 10.
- `similar_to` stays weak at every threshold. "Books like Gone Girl" needs knowledge about the book, and the query text alone does not carry that knowledge.

### Limitation of this evaluation

All the expected books in the evaluation set are well known. A popularity filter must look good on this set. To measure the cost of the filter fairly, the set needs some less-known books (1,000 to 10,000 ratings) as expected results.

### What I learned

- Popularity is a strong ranking signal for this dataset.
- A hard filter increases recall on popular books and removes most of the catalogue. Measure coverage together with recall.
- An evaluation set can be biased in the same direction as the change that you test. Check what the set rewards.

### Next step

Use popularity in the ranking instead of as a filter. Take the 200 nearest books by meaning, then re-rank them by similarity plus a weighted `log(num_ratings)`. Tune the weight with the evaluation set, after I add less-known books to it.
