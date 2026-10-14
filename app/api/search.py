from fastapi import APIRouter, Depends, Query
from sentence_transformers import SentenceTransformer
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import Book
from app.schemas.books import BookResponse

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


@router.get("/books", response_model=list[BookResponse])
def search_books(
    q: str = Query(min_length=1),
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    model = get_embedding_model()
    query_embedding = model.encode(q, normalize_embeddings=True).tolist()

    return db.scalars(
        select(Book).from_statement(
            text(
                """
                SELECT books.*
                FROM books
                JOIN book_embeddings
                    ON book_embeddings.book_id = books.id
                WHERE book_embeddings.embedding_model = :embedding_model
                    AND book_embeddings.embedding_dimension = :embedding_dimension
                ORDER BY book_embeddings.embedding <=> CAST(:query_embedding AS vector)
                LIMIT :limit
                """
            )
        ),
        {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "query_embedding": vector_literal(query_embedding),
            "limit": limit,
        },
    ).all()
