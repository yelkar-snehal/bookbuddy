from typing import Literal

from fastapi import APIRouter, Depends, Query
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.schemas.books import BookResponse
from app.schemas.search import BookSearchResult

router = APIRouter(prefix="/search", tags=["search"])

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
# Each lane returns this many candidates before fusion.
HYBRID_CANDIDATES = 50
# Standard Reciprocal Rank Fusion constant. A higher value gives
# lower-ranked candidates relatively more weight.
RRF_K = 60
_embedding_model: SentenceTransformer | None = None

BOOK_COLUMNS = """
    books.id,
    books.title,
    books.goodreads_id,
    books.url,
    books.author_id,
    books.star_rating,
    books.num_ratings,
    books.num_reviews
"""

# Books with few ratings are mostly study guides, summaries and obscure
# titles that crowd out the books readers want.
MIN_RATINGS_FILTER = """
    (
        CAST(:min_ratings AS integer) IS NULL
        OR books.num_ratings >= :min_ratings
    )
"""

SEMANTIC_SEARCH_SQL = f"""
    SELECT
        {BOOK_COLUMNS},
        book_embeddings.embedding <=> CAST(:query_embedding AS vector) AS distance,
        NULL AS score
    FROM books
    JOIN book_embeddings
        ON book_embeddings.book_id = books.id
    WHERE book_embeddings.embedding_model = :embedding_model
        AND book_embeddings.embedding_dimension = :embedding_dimension
        AND {MIN_RATINGS_FILTER}
    ORDER BY distance
    LIMIT :limit
"""

# Hybrid search runs two lanes and fuses them with Reciprocal Rank Fusion:
#   score = 1 / (RRF_K + semantic_rank) + 1 / (RRF_K + keyword_rank)
# A book found by only one lane gets only that lane's term.
#
# The keyword lane uses PostgreSQL full-text search on titles and author
# names. ts_rank is not BM25, but it follows the same idea: books that
# contain more of the query terms rank higher. The query terms are joined
# with OR, so "Dune by Frank Herbert" matches the title "Dune" and the
# author "Frank Herbert" separately.
HYBRID_SEARCH_SQL = f"""
    -- plainto_tsquery joins terms with AND. Replacing AND with OR and
    -- parsing the result with 'simple' keeps the english stems as they are,
    -- so the terms are not stemmed twice.
    WITH keyword_query AS (
        SELECT
            to_tsquery(
                'simple',
                replace(CAST(plainto_tsquery('english', :q) AS text), ' & ', ' | ')
            ) AS title_query,
            to_tsquery(
                'simple',
                replace(CAST(plainto_tsquery('simple', :q) AS text), ' & ', ' | ')
            ) AS author_query
    ),
    semantic_candidates AS (
        SELECT
            books.id,
            book_embeddings.embedding <=> CAST(:query_embedding AS vector) AS distance
        FROM books
        JOIN book_embeddings
            ON book_embeddings.book_id = books.id
        WHERE book_embeddings.embedding_model = :embedding_model
            AND book_embeddings.embedding_dimension = :embedding_dimension
            AND {MIN_RATINGS_FILTER}
        ORDER BY distance
        LIMIT :candidates
    ),
    semantic AS (
        SELECT
            id,
            distance,
            ROW_NUMBER() OVER (ORDER BY distance) AS rank
        FROM semantic_candidates
    ),
    -- Two separate lookups, so each one can use its own GIN index.
    keyword_matches AS (
        SELECT books.id
        FROM books, keyword_query
        WHERE to_tsvector('english', books.title) @@ keyword_query.title_query
        UNION
        SELECT books.id
        FROM authors
        JOIN books ON books.author_id = authors.id
        CROSS JOIN keyword_query
        WHERE to_tsvector('simple', authors.name) @@ keyword_query.author_query
    ),
    keyword_candidates AS (
        SELECT
            books.id,
            ts_rank(to_tsvector('english', books.title), keyword_query.title_query)
                + ts_rank(
                    to_tsvector('simple', coalesce(authors.name, '')),
                    keyword_query.author_query
                ) AS keyword_score
        FROM keyword_matches
        JOIN books ON books.id = keyword_matches.id
        LEFT JOIN authors ON authors.id = books.author_id
        CROSS JOIN keyword_query
        WHERE {MIN_RATINGS_FILTER}
        -- num_ratings breaks ties between editions with the same title.
        ORDER BY keyword_score DESC, books.num_ratings DESC NULLS LAST
        LIMIT :candidates
    ),
    keyword AS (
        SELECT
            id,
            ROW_NUMBER() OVER (
                ORDER BY keyword_score DESC
            ) AS rank
        FROM keyword_candidates
    ),
    fused AS (
        SELECT
            id,
            semantic.distance,
            coalesce(1.0 / (:rrf_k + semantic.rank), 0)
                + coalesce(1.0 / (:rrf_k + keyword.rank), 0) AS score
        FROM semantic
        FULL OUTER JOIN keyword USING (id)
    )
    SELECT
        {BOOK_COLUMNS},
        fused.distance,
        fused.score
    FROM fused
    JOIN books ON books.id = fused.id
    ORDER BY fused.score DESC
    LIMIT :limit
"""


def get_db():
    with SessionLocal() as session:
        yield session


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    return _embedding_model


def vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(value) for value in embedding) + "]"


@router.get("/books", response_model=list[BookSearchResult])
def search_books(
    q: str = Query(min_length=1),
    limit: int = Query(default=10, ge=1, le=50),
    min_ratings: int | None = Query(default=None, ge=0),
    mode: Literal["semantic", "hybrid"] = Query(default="semantic"),
    db: Session = Depends(get_db),
):
    model = get_embedding_model()
    query_embedding = model.encode(q, normalize_embeddings=True).tolist()

    sql = HYBRID_SEARCH_SQL if mode == "hybrid" else SEMANTIC_SEARCH_SQL

    rows = db.execute(
        text(sql),
        {
            "q": q,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "query_embedding": vector_literal(query_embedding),
            "limit": limit,
            "min_ratings": min_ratings,
            "candidates": HYBRID_CANDIDATES,
            "rrf_k": RRF_K,
        },
    ).mappings().all()

    return [
        BookSearchResult(
            book=BookResponse.model_validate(row),
            distance=row["distance"],
            score=row["score"],
        )
        for row in rows
    ]
