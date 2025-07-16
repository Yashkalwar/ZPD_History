from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

# Pydantic models for requests/responses
class LoginRequest(BaseModel):
    student_id: str
    password: Optional[str] = None

class LoginResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    message: str

class SessionResponse(BaseModel):
    logged_in: bool
    student_id: Optional[str] = None
    zpd_score: Optional[float] = None

# Dummy in-memory session store (replace with DB in production)
sessions = {}

@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest):
    # For demo, accept any student_id
    if not request.student_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Student ID required.")
    # Generate dummy token
    token = f"token_{request.student_id}"
    sessions[token] = {"student_id": request.student_id, "zpd_score": 2.5}
    return LoginResponse(success=True, token=token, message="Login successful.")

@router.post("/logout")
def logout(token: str):
    if token in sessions:
        del sessions[token]
        return {"success": True, "message": "Logged out."}
    return {"success": False, "message": "Invalid session."}

@router.get("/session", response_model=SessionResponse)
def get_session(token: str):
    session = sessions.get(token)
    if not session:
        return SessionResponse(logged_in=False)
    return SessionResponse(logged_in=True, student_id=session["student_id"], zpd_score=session["zpd_score"])
