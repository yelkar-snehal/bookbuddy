from pydantic import BaseModel

from app.schemas.books import BookResponse


class BookSearchResult(BaseModel):
    book: BookResponse
    distance: float
