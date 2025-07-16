# Centralized Pydantic models for requests/responses (for import across routers)
from pydantic import BaseModel
from typing import Optional, List

class LoginRequest(BaseModel):
    student_id: str
    password: Optional[str] = None

class LoginResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    message: str

class Chapter(BaseModel):
    id: str
    title: str
    start_page: int
    end_page: int

class QuestionRequest(BaseModel):
    chapter_id: str
    zpd_score: float
    previous_questions: Optional[List[str]] = []

class QuestionResponse(BaseModel):
    question: str
    answer: str
    difficulty: str
    focus_aspect: str

class ProgressResponse(BaseModel):
    chapters_completed: int
    total_questions: int
    zpd_score: float
