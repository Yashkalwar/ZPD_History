from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel
from typing import List

router = APIRouter()

# Dummy in-memory user/session store for demo
user_sessions = {}

class ZPDResponse(BaseModel):
    zpd_score: float

class ZPDUpdateRequest(BaseModel):
    token: str
    new_zpd: float

@router.get("/zpd", response_model=ZPDResponse)
def get_zpd(token: str, request: Request):
    from backend.routers.auth import sessions
    session = sessions.get(token)
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session.")
    return ZPDResponse(zpd_score=session["zpd_score"])

@router.post("/zpd", response_model=ZPDResponse)
def update_zpd(update: ZPDUpdateRequest, request: Request):
    from backend.routers.auth import sessions
    session = sessions.get(update.token)
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session.")
    if not (1.0 <= update.new_zpd <= 10.0):
        raise HTTPException(status_code=400, detail="ZPD must be between 1.0 and 10.0.")
    session["zpd_score"] = update.new_zpd
    return ZPDResponse(zpd_score=session["zpd_score"])

class HistoryItem(BaseModel):
    question: str
    user_answer: str
    correct: bool
    score: float
    feedback: str

class HistoryResponse(BaseModel):
    history: List[HistoryItem]

@router.get("/history", response_model=HistoryResponse)
def get_history(token: str, request: Request):
    if token not in user_sessions:
        return HistoryResponse(history=[])
    return HistoryResponse(history=user_sessions[token].get("history", []))

# Utility: Add to history (to be called from evaluate endpoint)
def add_to_history(token: str, item: HistoryItem):
    if token not in user_sessions:
        user_sessions[token] = {"history": []}
    user_sessions[token]["history"].append(item)
