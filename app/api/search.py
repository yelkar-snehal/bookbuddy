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
_embedding_model: SentenceTransformer | None = None


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
    db: Session = Depends(get_db),
):
    model = get_embedding_model()
    query_embedding = model.encode(q, normalize_embeddings=True).tolist()

    rows = db.execute(
        text(
            """
            SELECT
                books.id,
                books.title,
                books.goodreads_id,
                books.url,
                books.author_id,
                books.star_rating,
                books.num_ratings,
                books.num_reviews,
                book_embeddings.embedding <=> CAST(:query_embedding AS vector) AS distance
            FROM books
            JOIN book_embeddings
                ON book_embeddings.book_id = books.id
            WHERE book_embeddings.embedding_model = :embedding_model
                AND book_embeddings.embedding_dimension = :embedding_dimension
                -- Books with few ratings are mostly study guides, summaries
                -- and obscure titles that crowd out the books readers want.
                AND (
                    CAST(:min_ratings AS integer) IS NULL
                    OR books.num_ratings >= :min_ratings
                )
            ORDER BY distance
            LIMIT :limit
            """
        ),
        {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "query_embedding": vector_literal(query_embedding),
            "limit": limit,
            "min_ratings": min_ratings,
        },
    ).mappings().all()

    return [
        BookSearchResult(
            book=BookResponse.model_validate(row),
            distance=row["distance"],
        )
        for row in rows
    ]
