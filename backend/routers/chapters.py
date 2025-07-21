from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List

router = APIRouter()

class Chapter(BaseModel):
    id: str
    title: str
    start_page: int
    end_page: int



@router.get("/", response_model=List[Chapter])
def list_chapters(request: Request):
    """Return the chapter map loaded during startup."""
    return request.app.state.chapter_map

@router.get("/{chapter_id}", response_model=Chapter)
def get_chapter(chapter_id: str, request: Request):
    """Retrieve a single chapter from the loaded chapter map."""
    for chapter in request.app.state.chapter_map:
        if chapter["id"] == chapter_id:
            return chapter
    raise HTTPException(status_code=404, detail="Chapter not found.")
