# A-Level Study Assistant

An AI-powered study assistant that helps students prepare for A-Level exams by providing personalized explanations, practice questions, and feedback.

## Features
- AI-powered chat interface for asking questions about A-Level topics
- Context-aware responses using vector embeddings
- Practice question generation and evaluation
- User authentication and history tracking
- Interactive Streamlit frontend

## Setup

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy `env.example` to `.env` and fill in your OpenAI API key:
```bash
cp env.example .env
```

5. Initialize the database:
```bash
python scripts/init_db.py
```

6. Build the FAISS index:
```bash
python scripts/build_faiss.py
```

7. Run the application:
```bash
# Backend
uvicorn backend.app.main:app --reload

# Frontend
streamlit run frontend/app.py
```

## Project Structure
```
CODE_FINAL/
├── .gitignore
├── README.md
├── requirements.txt
├── env.example
├── data/
│   ├── raw/
│   └── faiss_index/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── models.py
│   │   ├── db.py
│   │   ├── vector_store.py
│   │   ├── endpoints/
│   │   │   ├── auth.py
│   │   │   ├── chat.py
│   │   │   └── evaluate.py
│   │   └── prompts.py
│   └── alembic/
├── frontend/
│   └── app.py
├── scripts/
│   ├── build_faiss.py
│   └── init_db.py
└── tests/
    ├── test_auth.py
    └── test_chat_flow.py
```

## Requirements
- Python 3.9+
- OpenAI API key
- PostgreSQL (optional, SQLite can be used for development)
- FAISS for vector storage
