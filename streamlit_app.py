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
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI

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
    if choice != "All Chapters":
        idx = [c["title"] for c in chapters].index(choice)
        selected_id = chapters[idx]["id"]
    else:
        selected_id = "all"

    if not VECTORSTORE_PATH.exists():
        docs = extract_text_with_metadata(PDF_PATH, chapters)
        chunks = split_documents(docs)
        create_and_save_vectorstore(chunks)

    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    retriever = load_retriever_and_reranker(
        embeddings, embeddings.query_instruction, selected_id
    )
    llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.7, max_tokens=1500)
    qa_chain = setup_qa_chain(llm, retriever)

    st.session_state["retriever"] = retriever
    st.session_state["llm"] = llm
    st.session_state["qa_chain"] = qa_chain
    st.session_state["asked"] = set()


def quiz():
    session = st.session_state["session"]
    llm = st.session_state["llm"]
    retriever = st.session_state["retriever"]
    chapter_title = st.session_state.get("selected_chapter", "All Chapters")
    asked = st.session_state["asked"]

    if "current_question" not in st.session_state:
        q, a, _ = generate_question_from_chapter_content(
            retriever=retriever,
            llm=llm,
            selected_chapter_title=chapter_title,
            previous_questions=asked,
            zpd_score=session.current_zpd,
        )
        st.session_state["current_question"] = q
        st.session_state["expected_answer"] = a
        asked.add(q.lower())

    st.write(f"**Question:** {st.session_state['current_question']}")
    answer = st.text_input("Your answer", key="answer")
    if st.button("Submit") and answer:
        feedback, correct, analysis = get_feedback_on_answer(
            user_answer=answer,
            expected_answer=st.session_state["expected_answer"],
            question=st.session_state["current_question"],
            llm=llm,
            context="",
            zpd_score=session.current_zpd,
        )
        old, new = student_mgr.update_student_zpd(
            student_session=session,
            is_correct=correct,
            is_partial=analysis.get("partially_correct", False),
        )
        st.write(feedback)
        if not correct and analysis.get("hint"):
            if st.button("Hint"):
                st.info(analysis["hint"])
        st.write(f"ZPD: {old:.1f} -> {new:.1f}")
        del st.session_state["current_question"]
        st.experimental_rerun()
    if st.button("New Question"):
        del st.session_state["current_question"]
        st.experimental_rerun()


step = st.session_state.get("login_step", "login")
if step == "login":
    login()
elif step == "register":
    register()
elif step == "select_chapter":
    select_chapter()
else:
    quiz()
