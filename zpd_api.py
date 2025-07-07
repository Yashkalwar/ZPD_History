from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import SessionLocal, engine, get_db
from models import User, UserZPD, Base
from ZPD_calculator import ZPDCalculator
from pydantic import BaseModel
from typing import List, Optional

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="ZPD Tracker API",
             description="API for tracking Zone of Proximal Development scores (1.0-10.0 scale)",
             version="1.0.0")

# Pydantic models
class ZPDResponse(BaseModel):
    zpd_score: float
    message: str

class ScoreUpdate(BaseModel):
    scores: List[float]  # List of scores (0.0 to 1.0)

# API Endpoints
@app.post("/users/{user_id}/update-scores", response_model=ZPDResponse)
def update_scores(
    user_id: int, 
    score_update: ScoreUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a user's ZPD score based on recent performance scores.
    
    - **user_id**: ID of the user
    - **scores**: List of recent scores (0.0 to 1.0, where 1.0 is perfect)
    """
    calculator = ZPDCalculator(db)
    try:
        zpd_score, message = calculator.update_user_zpd(user_id, score_update.scores)
        return {"zpd_score": zpd_score, "message": message}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/users/{user_id}/zpd", response_model=ZPDResponse)
def get_zpd(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a user's current ZPD score.
    
    - **user_id**: ID of the user
    """
    calculator = ZPDCalculator(db)
    try:
        zpd_score = calculator.get_user_zpd(user_id)
        return {
            "zpd_score": zpd_score,
            "message": f"Current ZPD score: {zpd_score:.1f}"
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("zpd_api:app", host="0.0.0.0", port=8000, reload=True)
