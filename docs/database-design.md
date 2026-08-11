# BookBuddy Database Design

## 1. Overview

Phase 1 uses PostgreSQL to store the normalized book metadata required by the application.

The schema is based on the characteristics discovered during dataset profiling rather than assuming a generic book schema.

The Phase 1 model contains two primary entities:

- `authors`
- `books`

Vector embeddings are intentionally excluded from Phase 1 and will be introduced as a separate model in Phase 2.

---

## 2. Entity Relationship Diagram

```text
┌──────────────────────────┐
│         AUTHORS          │
├──────────────────────────┤
│ id            PK         │
│ name                     │
│ num_books                │
│ num_followers            │
│ about                    │
└────────────┬─────────────┘
             │
             │ 1
             │
             │ N
┌────────────▼─────────────┐
│          BOOKS           │
├──────────────────────────┤
│ id            PK         │
│ goodreads_id             │
│ url           UNIQUE     │
│ title                    │
│ author_id     FK         │
│ summary                  │
│ star_rating              │
│ num_ratings              │
│ num_reviews              │
│ genres        JSONB      │
│ first_published          │
│ kindle_price             │
│ community_reviews JSONB  │
└──────────────────────────┘