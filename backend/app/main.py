from fastapi import FastAPI
from backend.routers import query
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Decision Intelligence Assistant")

app.include_router(query.router)

@app.get("/health")
def health():
    return {"status": "ok"}