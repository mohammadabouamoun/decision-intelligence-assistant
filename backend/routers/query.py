from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
from backend.utils.retrieval import retrieve_similar
from backend.utils.llm_client import rag_answer, non_rag_answer, zero_shot_priority
from backend.utils.ml_predictor import predict_priority
from backend.utils.logger import log_query

router = APIRouter(prefix="/query", tags=["query"])

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    question: str
    rag_answer: str
    non_rag_answer: str
    ml_priority: Dict[str, Any]
    llm_priority: Dict[str, Any]
    rag_latency_ms: float
    non_rag_latency_ms: float
    ml_latency_ms: float
    llm_priority_latency_ms: float
    rag_cost_usd: float
    non_rag_cost_usd: float
    llm_priority_cost_usd: float
    retrieved_chunks: List[Dict[str, Any]]

@router.post("/", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    retrieved = retrieve_similar(request.question, top_k=request.top_k)
    
    rag_text, rag_latency, rag_cost = rag_answer(request.question, retrieved)
    non_rag_text, non_rag_latency, non_rag_cost = non_rag_answer(request.question)
    
    ml_label, ml_conf, ml_latency = predict_priority(request.question)
    ml_priority = {"label": str(ml_label), "confidence": float(ml_conf)}
    
    llm_text, llm_latency, llm_cost, llm_priority = zero_shot_priority(request.question)
    
    response = QueryResponse(
        question=request.question,
        rag_answer=rag_text,
        non_rag_answer=non_rag_text,
        ml_priority=ml_priority,
        llm_priority=llm_priority,
        rag_latency_ms=float(rag_latency),
        non_rag_latency_ms=float(non_rag_latency),
        ml_latency_ms=float(ml_latency),
        llm_priority_latency_ms=float(llm_latency),
        rag_cost_usd=float(rag_cost),
        non_rag_cost_usd=float(non_rag_cost),
        llm_priority_cost_usd=float(llm_cost),
        retrieved_chunks=retrieved
    )
    
    # Log the query data for debugging and evaluation
    log_query({
        "question": request.question,
        "rag_answer": rag_text,
        "non_rag_answer": non_rag_text,
        "ml_priority": ml_priority,
        "llm_priority": llm_priority,
        "rag_latency_ms": rag_latency,
        "non_rag_latency_ms": non_rag_latency,
        "ml_latency_ms": ml_latency,
        "llm_priority_latency_ms": llm_latency,
        "rag_cost_usd": rag_cost,
        "non_rag_cost_usd": non_rag_cost,
        "llm_priority_cost_usd": llm_cost,
        "retrieved_chunks": retrieved
    })
    
    return response