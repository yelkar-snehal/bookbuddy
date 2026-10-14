from fastapi import FastAPI
from app.api import authors, books, search

app = FastAPI(title="BookBuddy")

app.include_router(authors.router)
app.include_router(books.router)
app.include_router(search.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
