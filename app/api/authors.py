from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import Author, Book
from app.schemas.authors import AuthorResponse
from app.schemas.books import BookResponse

router = APIRouter(prefix="/authors", tags=["authors"])


def get_db():
    with SessionLocal() as session:
        yield session


@router.get("/{author_id}", response_model=AuthorResponse)
def get_author(author_id: int, db: Session = Depends(get_db)):
    author = db.scalar(
        select(Author).where(Author.id == author_id)
    )

    if author is None:
        raise HTTPException(status_code=404, detail="Author not found")

    return author


@router.get("/{author_id}/books", response_model=list[BookResponse])
def list_author_books(author_id: int, db: Session = Depends(get_db)):
    author_exists = db.scalar(
        select(Author.id).where(Author.id == author_id)
    )

    if author_exists is None:
        raise HTTPException(status_code=404, detail="Author not found")

    return db.scalars(
        select(Book)
        .where(Book.author_id == author_id)
        .order_by(Book.id)
    ).all()
