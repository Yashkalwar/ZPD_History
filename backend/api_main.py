from fastapi import FastAPI
from backend.routers import auth, chapters, quiz, progress
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add project root to path
import main as core_logic

app = FastAPI(title="History Learning API")

@app.on_event("startup")
def startup_event():
    # Initialize core resources ONCE
    core_logic.check_environment()
    app.state.chapter_map = core_logic.load_chapter_map(core_logic.CHAPTER_MAP_PATH)
    embeddings = core_logic.HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True}
    )
    app.state.llm = core_logic.ChatOpenAI()
    app.state.retriever = core_logic.load_retriever_and_reranker(embeddings, query_instruction="history question")
    print("[Startup] LLM, retriever, and chapter map initialized.")

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(chapters.router, prefix="/chapters", tags=["chapters"])
app.include_router(quiz.router, prefix="/quiz", tags=["quiz"])
app.include_router(progress.router, prefix="/progress", tags=["progress"])

@app.get("/")
def root():
    return {"message": "History Learning API is running."}
