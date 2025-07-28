import streamlit as st

from student_manager import StudentManager
from main import (
    load_chapter_map,
    extract_text_with_metadata,
    split_documents,
    create_and_save_vectorstore,
    load_retriever_and_reranker,
    setup_qa_chain,
    generate_question_from_chapter_content,
    get_feedback_on_answer,
    VECTORSTORE_PATH,
    PDF_PATH,
    CHAPTER_MAP_PATH,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="History Tutor")
student_mgr = StudentManager()


def login():
    st.title("History Tutor")
    student_id = st.text_input("Student ID")
    if st.button("Login") and student_id:
        student = student_mgr.db.get_student(student_id)
        if student:
            session = student_mgr.create_session(
                student_id=student_id,
                student_name=student["student_name"],
                initial_zpd=student["zpd_score"],
            )
        else:
            st.session_state["register_id"] = student_id
            st.session_state["login_step"] = "register"
            return
        st.session_state["session"] = session
        st.session_state["login_step"] = "select_chapter"


def register():
    st.title("Register Student")
    name = st.text_input("Name")
    if st.button("Create") and name:
        student_id = st.session_state.get("register_id")
        student_mgr.db.add_student(student_id, name, 5.0)
        session = student_mgr.create_session(
            student_id=student_id,
            student_name=name,
            initial_zpd=5.0,
        )
        st.session_state["session"] = session
        st.session_state["login_step"] = "select_chapter"


def select_chapter():
    st.header(f"Welcome, {st.session_state['session'].student_name}")
    chapters = load_chapter_map(CHAPTER_MAP_PATH)
    chapter_titles = [c["title"] for c in chapters]
    options = ["All Chapters"] + chapter_titles
    choice = st.selectbox("Choose chapter", options)
    if st.button("Start"):
        st.session_state["selected_chapter"] = choice
        st.session_state["login_step"] = "quiz"
        initialize_qa(choice, chapters)


def initialize_qa(choice, chapters):
    try:
        # Get the selected chapter ID or use "all"
        selected_id = "all"
        if choice != "All Chapters":
            idx = [c["title"] for c in chapters].index(choice)
            selected_id = chapters[idx]["id"]
            print(f"Selected chapter ID: {selected_id}")

        # Ensure vector store exists
        if not VECTORSTORE_PATH.exists():
            print("Vector store not found. Creating a new one...")
            docs = extract_text_with_metadata(PDF_PATH, chapters)
            chunks = split_documents(docs)
            create_and_save_vectorstore(chunks)

        # Initialize embeddings with explicit device specification
        print("Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Initialize retriever with error handling
        print("Initializing retriever...")
        retriever = load_retriever_and_reranker(
            embeddings, 
            "",  # Empty query instruction as it's not used
            selected_id if selected_id != "all" else None
        )
        
        # Initialize LLM with proper error handling
        print("Initializing LLM...")
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1500,
            api_key=os.getenv("OPENAI_API_KEY"),
            request_timeout=30  # Add timeout to prevent hanging
        )
        
        # Initialize session state
        print("Initializing session state...")
        st.session_state.update({
            "retriever": retriever,
            "llm": llm,
            "asked": set(),
            "initialized": True,
            "selected_chapter_id": selected_id,
            "selected_chapter_title": choice
        })
        
        print("QA system initialized successfully!")
        return True
        
    except Exception as e:
        error_msg = f"Error initializing QA system: {str(e)}"
        print(error_msg)  # Log to console for debugging
        st.error(error_msg)
        if "retriever" in st.session_state:
            del st.session_state["retriever"]
        if "llm" in st.session_state:
            del st.session_state["llm"]
        st.stop()
        return False


def quiz():
    try:
        # Initialize session
        if "session" not in st.session_state:
            st.error("Session not found. Please start over from the login page.")
            st.stop()
            
        session = st.session_state["session"]
        st.title("History Quiz")
        st.write(f"Chapter: {st.session_state.get('selected_chapter', 'All Chapters')}")
        st.write(f"Your current ZPD score: {session.current_zpd:.1f}")

        # Initialize session state variables
        if "question" not in st.session_state:
            st.session_state.question = ""
        if "expected_answer" not in st.session_state:
            st.session_state.expected_answer = ""
        if "llm" not in st.session_state:
            st.session_state["llm"] = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1500,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        if "retriever" not in st.session_state:
            # Initialize retriever if not already done
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            chapter_map = load_chapter_map(CHAPTER_MAP_PATH)
            selected_chapter = st.session_state.get("selected_chapter", "All Chapters")
            selected_id = "all"
            if selected_chapter != "All Chapters":
                idx = [c["title"] for c in chapter_map].index(selected_chapter)
                selected_id = chapter_map[idx]["id"]
            st.session_state["retriever"] = load_retriever_and_reranker(
                embeddings, "", selected_id
            )
        if "asked" not in st.session_state:
            st.session_state["asked"] = set()
            
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        st.stop()

    # Generate a new question if needed
    if not st.session_state.get("current_question"):
        with st.spinner("Generating a new question..."):
            try:
                # Clear any previous error messages and feedback
                if "error_message" in st.session_state:
                    del st.session_state["error_message"]
                if "feedback" in st.session_state:
                    del st.session_state["feedback"]
                if "show_feedback" in st.session_state:
                    del st.session_state["show_feedback"]
                
                # Ensure we have a valid retriever and LLM
                if "retriever" not in st.session_state or "llm" not in st.session_state:
                    raise ValueError("Question generation components not properly initialized")
                
                # Get the selected chapter or use "All Chapters" as default
                selected_chapter = st.session_state.get("selected_chapter", "All Chapters")
                
                # Generate question using the predefined function
                q, a, difficulty = generate_question_from_chapter_content(
                    retriever=st.session_state["retriever"],
                    llm=st.session_state["llm"],
                    selected_chapter_title=selected_chapter,
                    previous_questions=st.session_state.get("asked", set()),
                    zpd_score=session.current_zpd,
                )
                
                if not q or not a:
                    raise ValueError("Failed to generate a valid question or answer")
                
                # Store question and expected answer in session state
                st.session_state["current_question"] = q
                st.session_state["expected_answer"] = a
                st.session_state["question_difficulty"] = difficulty
                
                # Initialize asked questions set if it doesn't exist
                if "asked" not in st.session_state:
                    st.session_state["asked"] = set()
                st.session_state["asked"].add(q.lower())
                
                # Reset feedback state
                st.session_state["show_feedback"] = False
                if "feedback" in st.session_state:
                    del st.session_state["feedback"]
                if "is_correct" in st.session_state:
                    del st.session_state["is_correct"]
                if "analysis" in st.session_state:
                    del st.session_state["analysis"]
                
                # Force a rerun to update the UI
                st.rerun()
                
            except Exception as e:
                error_msg = f"❌ Error generating question: {str(e)}"
                if "retriever" not in st.session_state:
                    error_msg += "\n\nRetriever not initialized. Please try selecting a chapter again."
                st.error(error_msg)
                st.session_state["error_message"] = error_msg
                
                # Add a button to retry question generation
                if st.button("Retry Generating Question"):
                    if "current_question" in st.session_state:
                        del st.session_state["current_question"]
                    if "error_message" in st.session_state:
                        del st.session_state["error_message"]
                    st.rerun()
                return

    # Display the current question with difficulty level
    difficulty = st.session_state.get("question_difficulty", "unknown")
    st.write(f"### Question ({difficulty.capitalize()} Level):")
    st.write(st.session_state["current_question"])
    
    # Show error message if any
    if "error_message" in st.session_state:
        st.error(st.session_state["error_message"])

    # Answer input with key to force clearing
    user_answer = st.text_area(
        "Your answer:", 
        key="user_answer",
        value=st.session_state.get("user_answer", "")
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Submit Answer") and user_answer:
            try:
                with st.spinner("Evaluating your answer..."):
                    # Get feedback on the answer
                    feedback, correct, analysis = get_feedback_on_answer(
                        user_answer=user_answer,
                        expected_answer=st.session_state["expected_answer"],
                        question=st.session_state["current_question"],
                        llm=st.session_state["llm"],
                        zpd_score=session.current_zpd
                    )
                    
                    # Update student's ZPD score
                    old, new = student_mgr.update_student_zpd(
                        student_session=session,
                        is_correct=correct,
                        is_partial=analysis.get("partially_correct", False),
                    )
                    
                    # Store feedback in session state
                    st.session_state.update({
                        "feedback": feedback,
                        "show_feedback": True,
                        "is_correct": correct,
                        "analysis": analysis,
                        "zpd_update": (old, new),
                        "show_hint": False  # Reset hint state for new feedback
                    })
                    
            except Exception as e:
                st.error(f"Error evaluating answer: {str(e)}")
            
            # Force a rerun to update the UI
            st.rerun()
    
    with col2:
        if st.button("New Question"):
            # Clear the current question to trigger generation of a new one
            if "current_question" in st.session_state:
                del st.session_state["current_question"]
            if "show_feedback" in st.session_state:
                del st.session_state["show_feedback"]
            st.rerun()
    
    # Display feedback if available
    if st.session_state.get("show_feedback", False):
        st.write("### Feedback:")
        st.write(st.session_state["feedback"])
        
        # Initialize show_hint in session state if not exists
        if "show_hint" not in st.session_state:
            st.session_state["show_hint"] = False
            
        # Show hint button if answer is incorrect and hint is available
        if not st.session_state.get("is_correct", False) and st.session_state.get("analysis", {}).get("hint"):
            if st.button("Show Hint"):
                st.session_state["show_hint"] = True
                st.rerun()
                
            # Display the hint if show_hint is True
            if st.session_state["show_hint"]:
                st.info(f"💡 **Hint:** {st.session_state['analysis']['hint']}")
        
            # Show ZPD score update if available
        if "zpd_update" in st.session_state:
            old, new = st.session_state["zpd_update"]
            st.write(f"**Your ZPD score updated:** {old:.1f} → {new:.1f}")
            
            # Add buttons for next question and show answer
            col1, col2 = st.columns(2)
            
            with col1:
                # Button to show the correct answer
                if st.button("Show Answer"):
                    st.session_state["show_answer"] = True
                    st.rerun()
                    
            with col2:
                # Button to continue to the next question
                if st.button("Next Question"):
                    # Clear relevant session state for next question
                    keys_to_clear = [
                        "current_question", "show_feedback", "show_hint", 
                        "feedback", "is_correct", "analysis", "zpd_update",
                        "user_answer", "show_answer"
                    ]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    # Force a rerun to update the UI with cleared state
                    st.rerun()
            
            # Display the correct answer if requested
            if st.session_state.get("show_answer", False):
                st.markdown("---")
                st.markdown("### Correct Answer:")
                st.info(st.session_state.get("expected_answer", "No answer available."))


step = st.session_state.get("login_step", "login")
if step == "login":
    login()
elif step == "register":
    register()
elif step == "select_chapter":
    select_chapter()
else:
    quiz()
