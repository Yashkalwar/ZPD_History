import traceback

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter()

class QuestionRequest(BaseModel):
    chapter_id: str
    zpd_score: float
    previous_questions: Optional[List[str]] = []

class QuestionResponse(BaseModel):
    question: str
    answer: str
    difficulty: str
    focus_aspect: str

@router.post("/question", response_model=QuestionResponse)
def get_question(request: QuestionRequest, fastapi_request: Request):
    try:
        # Validate chapter
        chapter_map = fastapi_request.app.state.chapter_map
        chapter = next((c for c in chapter_map if c["id"] == request.chapter_id), None)
        if not chapter:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chapter not found.")
        chapter_title = chapter["title"]

        # Get shared LLM and retriever
        llm = fastapi_request.app.state.llm
        retriever = fastapi_request.app.state.retriever
        previous_questions = set(request.previous_questions or [])

        # Import core logic
        import main as core_logic
        try:
            question, answer, difficulty = core_logic.generate_question_from_chapter_content(
                retriever, llm, request.zpd_score, chapter_title, previous_questions
            )
            print("***********")
            focus_aspect = "unknown"
            return QuestionResponse(
                question=question,
                answer=answer,  # Hide this in frontend if needed
                difficulty=difficulty,
                focus_aspect=focus_aspect
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating question: {str(e)}")
    except:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating question: {str(e)}")
# --- Evaluate Endpoint ---
from fastapi import Body
from backend.routers.progress import add_to_history, HistoryItem
from backend.routers.auth import sessions

class EvaluateRequest(BaseModel):
    question: str
    user_answer: str
    expected_answer: str
    zpd_score: float
    token: str

class EvaluateResponse(BaseModel):
    is_correct: bool
    score: float
    feedback: str
    hint: str

@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate_answer(request: EvaluateRequest, fastapi_request: Request):
    import main as core_logic
    llm = fastapi_request.app.state.llm
    try:
        # Validate inputs
        if not request.question or not request.user_answer or not request.expected_answer:
            raise HTTPException(status_code=400, detail="Missing required fields.")
        analysis = core_logic.analyze_student_answer(
            request.question,
            request.user_answer,
            request.expected_answer,
            llm,
            request.zpd_score,
        )

        # Record the attempt if a token was provided
        if request.token:
            add_to_history(
                request.token,
                HistoryItem(
                    question=request.question,
                    user_answer=request.user_answer,
                    correct=analysis["is_correct"],
                    score=analysis["score"],
                    feedback=analysis["feedback"],
                ),
            )

        return EvaluateResponse(
            is_correct=analysis["is_correct"],
            score=analysis["score"],
            feedback=analysis["feedback"],
            hint=analysis.get("hint", ""),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating answer: {str(e)}")

# --- Hint Endpoint ---
class HintRequest(BaseModel):
    question: str
    expected_answer: str
    zpd_score: float

class HintResponse(BaseModel):
    hint: str

@router.post("/hint", response_model=HintResponse)
def get_hint(request: HintRequest, fastapi_request: Request):
    import main as core_logic
    llm = fastapi_request.app.state.llm
    try:
        if not request.question or not request.expected_answer:
            raise HTTPException(status_code=400, detail="Missing required fields.")
        hint = core_logic.generate_hint(request.question, request.expected_answer, request.zpd_score, llm)
        return HintResponse(hint=hint)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating hint: {str(e)}")
 