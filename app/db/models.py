from sqlalchemy import BigInteger, ForeignKey, Identity, Integer, Numeric, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Author(Base):
    __tablename__ = "authors"

    id: Mapped[int] = mapped_column(
    BigInteger,
    Identity(),
    primary_key=True,
)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    num_books: Mapped[int | None] = mapped_column(Integer)
    num_followers: Mapped[int | None] = mapped_column(Integer)
    about: Mapped[str | None] = mapped_column(Text)

    books: Mapped[list["Book"]] = relationship(back_populates="author")


class Book(Base):
    __tablename__ = "books"

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
    )
    goodreads_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    url: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)

    author_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("authors.id"),
        index=True,
    )

    summary: Mapped[str | None] = mapped_column(Text)
    star_rating: Mapped[float | None] = mapped_column(Numeric(3, 2))
    num_ratings: Mapped[int | None] = mapped_column(Integer)
    num_reviews: Mapped[int | None] = mapped_column(Integer)

    genres: Mapped[dict | list | None] = mapped_column(JSONB)
    first_published: Mapped[str | None] = mapped_column(Text)
    kindle_price: Mapped[float | None] = mapped_column(Numeric(10, 2))
    community_reviews: Mapped[dict | None] = mapped_column(JSONB)

    author: Mapped[Author | None] = relationship(back_populates="books")