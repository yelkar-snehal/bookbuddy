from pydantic import BaseModel

from app.schemas.books import BookResponse


class BookSearchResult(BaseModel):
    book: BookResponse
    # Cosine distance from the query. None when only the keyword lane
    # found the book in hybrid search.
    distance: float | None
    # Reciprocal Rank Fusion score in hybrid search. None in semantic search.
    score: float | None = None
