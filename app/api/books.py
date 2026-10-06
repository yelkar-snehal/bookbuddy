from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import Book
from app.schemas.books import BookResponse

router = APIRouter(prefix="/books", tags=["books"])


def get_db():
    with SessionLocal() as session:
        yield session


@router.get("", response_model=list[BookResponse])
def list_books(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    return db.scalars(
        select(Book)
        .order_by(Book.id)
        .limit(limit)
        .offset(offset)
    ).all()


@router.get("/{book_id}", response_model=BookResponse)
def get_book(book_id: int, db: Session = Depends(get_db)):
    book = db.scalar(
        select(Book).where(Book.id == book_id)
    )

    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")

    return book
