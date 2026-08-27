from fastapi import FastAPI
from app.api import books

app = FastAPI(title="BookBuddy")

app.include_router(books.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}