from pydantic import BaseModel, ConfigDict


class BookResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    goodreads_id: str
    url: str
    author_id: int | None
    star_rating: float | None
    num_ratings: int | None
    num_reviews: int | None