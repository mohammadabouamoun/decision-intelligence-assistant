from fastapi import FastAPI
from backend.routers import query
from backend.utils.retrieval import init_retrieval
from backend.utils.ml_predictor import init_ml
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()

app = FastAPI(title="Decision Intelligence Assistant")

@app.on_event("startup")
async def startup_event():
    init_retrieval()
    init_ml()
    print("All components initialized.")

app.include_router(query.router)

@app.get("/health")
def health():
    return {"status": "ok"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 