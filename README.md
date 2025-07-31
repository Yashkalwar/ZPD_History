# 📚 A-Level History Study Buddy

Welcome to your personal AI-powered study assistant for A-Level History! This tool helps you study smarter by adapting to your learning pace and style. Whether you're cramming for exams or just want to understand historical events better, I'm here to help.

## ✨ What Makes This Special?

- **Adaptive Learning**: The system adjusts question difficulty based on your performance, just like a real tutor would
- **Interactive Quizzes**: Test your knowledge with automatically generated questions
- **Instant Feedback**: Get detailed explanations and feedback on your answers
- **Chapter-Specific Practice**: Focus on specific historical periods or topics
- **Progress Tracking**: Watch your knowledge grow over time

## 🚀 Quick Start Guide

### Prerequisites
Before we begin, make sure you have:
- Python 3.8 or higher
- An OpenAI API key (get one at [OpenAI's website](https://platform.openai.com/))
- Basic knowledge of command line/terminal

### Step 1: Get the Code
First, clone this repository to your computer:
```bash
git clone https://github.com/your-username/alevel-history-assistant.git
cd alevel-history-assistant
```

### Step 2: Set Up Your Environment

1. **Create a virtual environment** (keeps your project dependencies organized):
   ```bash
   # On Windows:
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**:
   - Create a new file called `.env` in the project folder
   - Add your OpenAI API key like this:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```

### Step 3: Start Learning!

#### Option 1: Command Line Interface (For Quick Use)
```bash
python main.py
```

#### Option 2: Web Interface (More Interactive)

1. **Initialize the Database** (run once):
   ```bash
   # In the project directory
   python create_sample_data.py
   ```
   This will set up the initial database with sample data

2. **Start the Backend API** (in a terminal):
   ```bash
   # In the project directory
   uvicorn quiz_api:app --reload
   ```
   This will start the FastAPI backend with auto-reload enabled

2. **Start the Frontend** (in a new terminal):
   ```bash
   # In the project directory
   streamlit run streamlit_frontend.py
   ```
   This will automatically open your browser to `http://localhost:8501`

> **Note**: Make sure both the backend and frontend are running for full functionality. The frontend depends on the backend API to work properly.
