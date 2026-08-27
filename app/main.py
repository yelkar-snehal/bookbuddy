from fastapi import FastAPI

app = FastAPI(title="BookBuddy")


@app.get("/health")
def health_check():
    return {"status": "ok"}