from pydantic import BaseModel, ConfigDict


class AuthorResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    num_books: int | None
    num_followers: int | None
    about: str | None
