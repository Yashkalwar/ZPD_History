from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

router = APIRouter()

class Chapter(BaseModel):
    id: str
    title: str
    start_page: int
    end_page: int



@router.get("/", response_model=List[Chapter])
def list_chapters():
    return CHAPTERS

@router.get("/{chapter_id}", response_model=Chapter)
def get_chapter(chapter_id: str):
    for chapter in CHAPTERS:
        if chapter["id"] == chapter_id:
            return chapter
    raise HTTPException(status_code=404, detail="Chapter not found.")
