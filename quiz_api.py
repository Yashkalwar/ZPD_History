from typing import Optional, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from student_manager import StudentManager, SessionExpiredError
from main import (
    load_chapter_map,
    extract_text_with_metadata,
    split_documents,
    create_and_save_vectorstore,
    load_retriever_and_reranker,
    generate_question_from_chapter_content,
    get_feedback_on_answer,
    VECTORSTORE_PATH,
    PDF_PATH,
    CHAPTER_MAP_PATH,
)
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI

app = FastAPI(title="Quiz API")

student_mgr = StudentManager()

# Global resources
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.7, max_tokens=1500)
retrievers: Dict[str, any] = {}


class LoginRequest(BaseModel):
    student_id: str
    name: Optional[str] = None


class QuestionRequest(BaseModel):
    student_id: str
    chapter_title: str


class AnswerRequest(BaseModel):
    student_id: str
    question: str
    expected_answer: str
    user_answer: str


def ensure_vectorstore():
    if not VECTORSTORE_PATH.exists():
        chapters = load_chapter_map(CHAPTER_MAP_PATH)
        docs = extract_text_with_metadata(PDF_PATH, chapters)
        chunks = split_documents(docs)
        create_and_save_vectorstore(chunks)


def get_retriever(chapter_id: str):
    if chapter_id not in retrievers:
        ensure_vectorstore()
        retrievers[chapter_id] = load_retriever_and_reranker(
            embeddings,
            embeddings.query_instruction,
            chapter_id,
        )
    return retrievers[chapter_id]


@app.post("/login")
def login(req: LoginRequest):
    student = student_mgr.db.get_student(req.student_id)
    if student:
        session = student_mgr.create_session(
            student_id=req.student_id,
            student_name=student["student_name"],
            initial_zpd=student["zpd_score"],
        )
    else:
        if not req.name:
            raise HTTPException(400, "Student not found. Provide name to register.")
        student_mgr.db.add_student(req.student_id, req.name, 5.0)
        session = student_mgr.create_session(
            student_id=req.student_id,
            student_name=req.name,
            initial_zpd=5.0,
        )
    return {"student_name": session.student_name, "zpd": session.current_zpd}


@app.get("/chapters")
def chapters():
    return load_chapter_map(CHAPTER_MAP_PATH)


@app.post("/generate-question")
def generate_question(req: QuestionRequest):
    session = student_mgr.get_session(req.student_id)
    if not session:
        raise HTTPException(401, "Invalid or expired session")
    chapter_map = load_chapter_map(CHAPTER_MAP_PATH)
    chapter_id = "all"
    for c in chapter_map:
        if c["title"] == req.chapter_title:
            chapter_id = c["id"]
            break
    retriever = get_retriever(chapter_id)
    question, answer, _ = generate_question_from_chapter_content(
        retriever=retriever,
        llm=llm,
        selected_chapter_title=req.chapter_title,
        previous_questions=set(),
        zpd_score=session.current_zpd,
    )
    return {"question": question, "expected_answer": answer}


@app.post("/submit-answer")
def submit_answer(req: AnswerRequest):
    session = student_mgr.get_session(req.student_id)
    if not session:
        raise HTTPException(401, "Invalid or expired session")
    feedback, correct, analysis = get_feedback_on_answer(
        user_answer=req.user_answer,
        expected_answer=req.expected_answer,
        question=req.question,
        llm=llm,
        context="",
        zpd_score=session.current_zpd,
    )
    try:
        old, new = student_mgr.update_student_zpd(
            student_session=session,
            is_correct=correct,
            is_partial=analysis.get("partially_correct", False),
        )
    except SessionExpiredError:
        raise HTTPException(401, "Session expired. Please log in again.")
    return {
        "feedback": feedback,
        "correct": correct,
        "hint": analysis.get("hint"),
        "old_zpd": old,
        "new_zpd": new,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("quiz_api:app", host="0.0.0.0", port=8000)
