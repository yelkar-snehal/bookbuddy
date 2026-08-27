from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import Book
from app.schemas.books import BookResponse

router = APIRouter(prefix="/books", tags=["books"])


def get_db():
    with SessionLocal() as session:
        yield session


@router.get("/{book_id}", response_model=BookResponse)
def get_book(book_id: int, db: Session = Depends(get_db)):
    book = db.scalar(
        select(Book).where(Book.id == book_id)
    )

    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")

    return book