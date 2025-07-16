import streamlit as st
import requests
import json
import os

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'login'
API_URL = "http://localhost:8000"

st.set_page_config(page_title="History Tutor", page_icon="📚", layout="centered")

# Initialize session state
if "student_id" not in st.session_state:
    st.session_state.student_id = None
if "student_name" not in st.session_state:
    st.session_state.student_name = None
if "zpd_score" not in st.session_state:
    st.session_state.zpd_score = 2.5  # Default ZPD score
if "chapter_id" not in st.session_state:
    st.session_state.chapter_id = None
if "chapter_title" not in st.session_state:
    st.session_state.chapter_title = None
if "page" not in st.session_state:
    st.session_state.page = "login"

# --- Login Page ---
def login_page():
    st.title("📚 History Tutor Login")
    student_id = st.text_input("Student ID")
    if st.button("Login"):
        if not student_id:
            st.error("Please enter your Student ID.")
            return
        try:
            resp = requests.post(f"{API_URL}/api/students/login", json={"student_id": student_id})
            data = resp.json()
            
            if resp.status_code == 200 and data.get("success"):
                # Update session with backend response data
                st.session_state.student_id = student_id
                st.session_state.student_name = data.get("student_name", f"Student {student_id}")
                st.session_state.zpd_score = float(data.get("zpd_score", 2.5))
                st.session_state.page = "chapter_select"
                st.rerun()
            else:
                # Show error details from backend
                error_msg = data.get("detail") or data.get("message", "Login failed. Please try again.")
                st.error(f"Login failed: {error_msg}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {str(e)}")
        except json.JSONDecodeError:
            st.error("Error: Invalid response from server. Please try again.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

# --- Chapter Selection Page ---
def load_chapter_data():
    """Load chapter data from the backend API or fallback to local file."""
    try:
        # Try to load from backend API first
        resp = requests.get(f"{API_URL}/chapters")
        if resp.status_code == 200:
            return resp.json()
            
        # If API fails, try to load from local file
        try:
            with open("data/raw/chapter_map.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            st.error("Chapter data file not found")
            return None
            
    except Exception as e:
        st.error(f"Error loading chapter data: {str(e)}")
        return None

def chapter_select_page():
    st.title("Select a Chapter")
    
    # Load chapter data
    chapters = load_chapter_data()
    
    if not chapters or not isinstance(chapters, list):
        st.error("No valid chapter data available. Please try again later.")
        return
    
    # Sort chapters by their ID to maintain consistent order
    chapters = sorted(chapters, key=lambda x: x.get('id', ''))
    
    # Create chapter display strings
    chapter_titles = []
    chapter_data = []
    
    for chapter in chapters:
        try:
            chapter_id = chapter.get('id', '').strip()
            title = chapter.get('title', 'Untitled').strip()
            start = int(chapter.get('start_page', 0))
            end = int(chapter.get('end_page', 0))
            
            if not chapter_id or not title:
                continue
                
            # Format the display text
            display_text = title
            if start > 0 and end > 0:
                display_text += f" (Pages {start}-{end})"
            
            chapter_titles.append(display_text)
            chapter_data.append({
                'id': chapter_id,
                'title': title,
                'start_page': start,
                'end_page': end
            })
            
        except Exception as e:
            st.warning(f"Skipping invalid chapter: {str(e)}")
            continue
    
    if not chapter_titles:
        st.error("No valid chapters found")
        return
    
    # Create the chapter selection UI
    st.write("### Available Chapters")
    
    # Create a row for each chapter with a button
    for i, (title, data) in enumerate(zip(chapter_titles, chapter_data)):
        # Use columns to create a card-like layout
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{title}**")
        with col2:
            if st.button("Start Quiz", key=f"start_quiz_{i}"):
                # Ensure chapter_id is in the format 'chapter_X'
                chapter_id = str(data['id']).strip()
                if not chapter_id.startswith('chapter_'):
                    # Extract number from ID and format as 'chapter_X'
                    chapter_num = ''.join(filter(str.isdigit, chapter_id)) or '1'
                    chapter_id = f"chapter_{chapter_num}"
                st.session_state.chapter_id = chapter_id
                st.session_state.chapter_title = data['title']
                st.session_state.page = "quiz"
                st.rerun()
        
        # Add a separator between chapters
        st.markdown("---")
    
    # Add a debug section (collapsed by default)
    with st.expander("Debug Information"):
        st.write("### Chapter Data")
        st.json(chapter_data)
        st.write("### Session State")
        st.json({
            "chapter_id": st.session_state.get("chapter_id"),
            "chapter_title": st.session_state.get("chapter_title"),
            "student_id": st.session_state.get("student_id"),
            "zpd_score": st.session_state.get("zpd_score")
        })
    
    # Logout button
    if st.sidebar.button("Logout", key="logout_button"):
        try:
            # Clear the session
            st.session_state.page = "login"
            st.session_state.student_id = None
            st.session_state.student_name = None
            st.session_state.zpd_score = 2.5
            st.session_state.chapter_id = None
            st.session_state.chapter_title = None
            st.rerun()
        except Exception as e:
            st.error(f"Error logging out: {str(e)}")

# --- Quiz Page ---
def quiz_page():
    st.title(f"Quiz: {st.session_state.chapter_title}")
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "current_answer" not in st.session_state:
        st.session_state.current_answer = None
    if "current_difficulty" not in st.session_state:
        st.session_state.current_difficulty = None
    if "current_expected_answer" not in st.session_state:
        st.session_state.current_expected_answer = None
    if "current_hint" not in st.session_state:
        st.session_state.current_hint = None
    if "previous_questions" not in st.session_state:
        st.session_state.previous_questions = []
    if "feedback" not in st.session_state:
        st.session_state.feedback = None
    if "answered" not in st.session_state:
        st.session_state.answered = False

    # Fetch a question if needed
    if st.session_state.current_question is None or st.session_state.answered:
        try:
            # Prepare the request payload according to the backend's QuestionRequest model
            payload = {
                "chapter_id": st.session_state.chapter_id,
                "zpd_score": float(st.session_state.zpd_score or 2.5),
                "previous_questions": st.session_state.previous_questions or []
            }
            resp = requests.post(f"{API_URL}/api/quiz/question", json=payload)
            
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.current_question = data["question"]
                st.session_state.current_expected_answer = data["answer"]
                st.session_state.current_difficulty = data.get("difficulty", "medium").title()
                st.session_state.current_focus_aspect = data.get("focus_aspect", "general")
                st.session_state.current_hint = None
                st.session_state.feedback = None
                st.session_state.answered = False
                
                # Add the question to previous_questions to avoid repeats
                if st.session_state.current_question not in st.session_state.previous_questions:
                    st.session_state.previous_questions.append(st.session_state.current_question)
            else:
                error_detail = resp.json().get("detail", "No error details provided")
                st.error(f"Failed to fetch question: {error_detail}")
                st.json(resp.json())  # Show full error response for debugging
                return
                
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {str(e)}")
            return
        except json.JSONDecodeError:
            st.error("Invalid response from server. Please try again.")
            return
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return

    st.markdown(f"**Difficulty:** {st.session_state.current_difficulty.title()}")
    st.markdown(f"**Question:** {st.session_state.current_question}")
    answer = st.text_area("Your Answer", value=st.session_state.current_answer or "", key="answer_input")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit Answer") and not st.session_state.answered:
            if not answer.strip():
                st.warning("Please enter an answer before submitting.")
            else:
                try:
                    eval_payload = {
                        "question": st.session_state.current_question,
                        "user_answer": answer,
                        "expected_answer": st.session_state.current_expected_answer,
                        "zpd_score": st.session_state.zpd_score or 2.5,
                        "token": st.session_state.token
                    }
                    eval_resp = requests.post(f"{API_URL}/quiz/evaluate", json=eval_payload)
                    if eval_resp.status_code == 200:
                        eval_data = eval_resp.json()
                        st.session_state.feedback = eval_data["feedback"]
                        st.session_state.current_hint = eval_data.get("hint")
                        st.session_state.answered = True
                        st.session_state.previous_questions.append(st.session_state.current_question)
                        # Update ZPD if provided
                        if "zpd_score" in eval_data:
                            st.session_state.zpd_score = eval_data["zpd_score"]
                    else:
                        st.error(eval_resp.json().get("detail", "Failed to evaluate answer."))
                except Exception as e:
                    st.error(f"Error evaluating answer: {str(e)}")
    with col2:
        if st.button("Get Hint") and not st.session_state.current_hint:
            try:
                hint_payload = {
                    "question": st.session_state.current_question,
                    "expected_answer": st.session_state.current_expected_answer,
                    "zpd_score": st.session_state.zpd_score or 2.5,
                    "token": st.session_state.token
                }
                hint_resp = requests.post(f"{API_URL}/quiz/hint", json=hint_payload)
                if hint_resp.status_code == 200:
                    st.session_state.current_hint = hint_resp.json()["hint"]
                else:
                    st.error(hint_resp.json().get("detail", "Failed to fetch hint."))
            except Exception as e:
                st.error(f"Error fetching hint: {str(e)}")

    if st.session_state.feedback:
        st.info(st.session_state.feedback)
    if st.session_state.current_hint:
        st.info(f"💡 Hint: {st.session_state.current_hint}")

    if st.session_state.answered:
        if st.button("Next Question"):
            st.session_state.current_question = None
            st.session_state.current_answer = None
            st.session_state.current_difficulty = None
            st.session_state.current_expected_answer = None
            st.session_state.current_hint = None
            st.session_state.feedback = None
            st.session_state.answered = False
            st.rerun()
        if st.button("Quit Quiz"):
            st.session_state.page = "chapter_select"
            st.session_state.current_question = None
            st.session_state.current_answer = None
            st.session_state.current_difficulty = None
            st.session_state.current_expected_answer = None
            st.session_state.current_hint = None
            st.session_state.feedback = None
            st.session_state.answered = False
            st.rerun()

# --- Progress/History Page ---
def progress_page():
    st.title("📈 Your Progress & History")
    try:
        # Show current ZPD score from session state
        if st.session_state.zpd_score is not None:
            st.markdown(f"**Current ZPD Score:** {st.session_state.zpd_score:.2f}")
        
        # Initialize empty history (since we don't have a backend for this yet)
        history = []
        st.info("Quiz history tracking will be available soon.")
        if history:
            st.markdown("### Answer History:")
            for item in history[::-1]:
                st.markdown(f"- **Q:** {item['question']}\n    - **Your Answer:** {item['user_answer']}\n    - **Correct:** {'✅' if item['correct'] else '❌'}\n    - **Score:** {item['score']}\n    - **Feedback:** {item['feedback']}")
        else:
            st.info("No quiz history yet. Try some questions!")
        if st.button("Back to Quiz"):
            st.session_state.page = "quiz"
            st.rerun()
        if st.button("Logout"):
            try:
                resp = requests.post(f"{API_URL}/api/students/logout", json={"token": st.session_state.token})
                if resp.status_code == 200:
                    st.session_state.page = "login"
                    st.session_state.token = None
                    st.session_state.student_id = None
                    st.rerun()
                else:
                    st.error(resp.json().get("message", "Logout failed"))
            except Exception as e:
                st.error(f"Error logging out: {str(e)}")
    except Exception as e:
        st.error(f"Error loading progress: {str(e)}")

# --- Navigation Sidebar ---
st.sidebar.title("Navigation")
if st.session_state.page not in ["login"]:
    if st.sidebar.button("Quiz"):
        st.session_state.page = "quiz"
        st.rerun()
    if st.sidebar.button("Progress/History"):
        st.session_state.page = "progress"
        st.rerun()
    if st.sidebar.button("Chapter Selection"):
        st.session_state.page = "chapter_select"
        st.rerun()
    st.sidebar.markdown(f"---\n**User:** {st.session_state.student_name or st.session_state.student_id or ''}")

# --- Page Routing ---
if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "chapter_select":
    chapter_select_page()
elif st.session_state.page == "quiz":
    quiz_page()
elif st.session_state.page == "progress":
    progress_page()
