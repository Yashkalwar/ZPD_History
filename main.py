"""
ZPD-Based Adaptive Learning System

This module implements an adaptive learning system that uses the Zone of Proximal Development (ZPD)
to personalize the learning experience. It features:
- Document processing and question generation
- Adaptive difficulty based on student performance
- Interactive Q&A with feedback
- Vector-based semantic search using FAISS

External Resources and Credits:
- PyMuPDF (fitz): For PDF text extraction (https://pypi.org/project/PyMuPDF/)
- LangChain: For document processing and LLM integration (https://python.langchain.com/)
- FAISS: For efficient vector similarity search (https://github.com/facebookresearch/faiss)
- HuggingFace: For embeddings and cross-encoders (https://huggingface.co/)
- OpenAI: For GPT language model integration (https://openai.com/)
- FuzzyWuzzy: For string matching and similarity (https://github.com/seatgeek/fuzzywuzzy)
- BGE Embeddings: For document embeddings (https://huggingface.co/BAAI/bge-base-en-v1.5)
- BGE Reranker: For document re-ranking (https://huggingface.co/BAAI/bge-reranker-base)

Note: This implementation is based on educational research in adaptive learning systems
and makes use of several open-source libraries. All external code is used in accordance
with their respective licenses.
"""

# Standard library imports
import os
import sys
import re
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any, Union

# Third-party imports
import fitz  # PyMuPDF - For PDF processing: https://pypi.org/project/PyMuPDF/
from dotenv import load_dotenv  # For loading environment variables
from fuzzywuzzy import fuzz  # For string matching: https://github.com/seatgeek/fuzzywuzzy

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS  # FAISS for vector search: https://github.com/facebookresearch/faiss
from langchain_community.embeddings import HuggingFaceBgeEmbeddings  # BERT-based embeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # OpenAI's chat models
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.messages import HumanMessage, SystemMessage

# Local imports
from ZPD_calculator import ZPDCalculator  # Custom module for ZPD calculations

# Load environment variables
load_dotenv()

# --- Application Constants ---
# These paths are relative to the project root directory
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "raw" / "history.pdf"  # Path to the source PDF
VECTORSTORE_PATH = BASE_DIR / "data" / "faiss_index_optimized"  # Where FAISS index will be stored
CHAPTER_MAP_PATH = BASE_DIR / "data" / "raw" / "chapter_map.json"  # Chapter metadata

# --- Utility Functions ---
def check_environment():
    """Checks for the OpenAI API key in the environment variables."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a '.env' file and add your key: OPENENAI_API_KEY='your_key_here'")
        sys.exit(1)
    print("OpenAI API key found.")

def load_chapter_map(map_path: Path) -> list[dict]:
    """Loads the chapter mapping from a JSON file."""
    if not map_path.exists():
        raise FileNotFoundError(f"Error: Chapter map file not found at {map_path}")
    try:
        with open(map_path, 'r', encoding='utf-8') as f:
            chapter_map = json.load(f)
            print(f"Loaded {len(chapter_map)} chapters from {map_path}")
            return chapter_map
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from chapter map file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading chapter map: {e}")
        sys.exit(1)

def extract_text_with_metadata(pdf_path: Path, chapter_map: list[dict]) -> list[Document]:
    """
    Extracts text and page numbers from a PDF using PyMuPDF (fitz) and enriches metadata with chapter info.
    Each page's text is stored as a LangChain Document with metadata.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"Error: PDF file not found at {pdf_path}")

    print(f"Reading PDF from: {pdf_path} using PyMuPDF...")
    docs = []
    try:
        with fitz.open(pdf_path) as pdf_document:
            print(f"Found {len(pdf_document)} pages in the PDF.")
            for page_num, page in enumerate(pdf_document, start=1):
                text = re.sub(r'\s+', ' ', page.get_text()).strip()
                current_chapter_id = "unknown"
                current_chapter_title = "Unknown Chapter"
                for chapter_info in chapter_map:
                    if chapter_info["start_page"] <= page_num <= chapter_info["end_page"]:
                        current_chapter_id = chapter_info["id"]
                        current_chapter_title = chapter_info["title"]
                        break
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": str(pdf_path.name),
                            "page": page_num,
                            "chapter_id": current_chapter_id,
                            "chapter_title": current_chapter_title
                        }
                    ))
            print(f"Successfully extracted text from {len(docs)} pages.")
            return docs
    except Exception as e:
        print(f"Error processing PDF with PyMuPDF: {e}")
        sys.exit(1)

def split_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """Splits a list of Documents into smaller chunks while preserving metadata."""
    print(f"Splitting {len(docs)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""], length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks

def create_and_save_vectorstore(chunks: list[Document]):
    """
    Creates and saves a FAISS (Facebook AI Similarity Search) vector store from document chunks.
    
    This function uses BGE (BAAI General Embedding) model to create dense vector representations
    of the text chunks and stores them in a FAISS index for efficient similarity search.
    
    References:
    - FAISS: https://github.com/facebookresearch/faiss
    - BGE Embeddings: https://huggingface.co/BAAI/bge-base-en-v1.5
    """
    try:
        print("Initializing BGE embeddings for vector store creation...")
        # Using BGE (BAAI General Embedding) model for creating document embeddings
        # Model card: https://huggingface.co/BAAI/bge-base-en-v1.5
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},  # Using CPU for compatibility
            encode_kwargs={'normalize_embeddings': True}  # Normalize embeddings for cosine similarity
        )
        
        # Create FAISS index from document chunks
        # FAISS provides efficient similarity search and clustering of dense vectors
        print(f"Creating FAISS index from {len(chunks)} chunks...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Save the vector store for later use
        VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(VECTORSTORE_PATH))
        print(f"✅ Vector store created and saved successfully at: {VECTORSTORE_PATH}")
        
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        sys.exit(1)

def load_retriever_and_reranker(embeddings_model, query_instruction: str, selected_chapter_id: str = None):
    """
    Loads the vector store and sets up a sophisticated retriever with a re-ranking stage.
    
    This function combines FAISS for efficient vector similarity search with a cross-encoder
    re-ranker to improve retrieval quality. The re-ranker uses BGE-Reranker to reorder the
    initial results based on more sophisticated semantic understanding.
    
    References:
    - FAISS: https://github.com/facebookresearch/faiss
    - BGE-Reranker: https://huggingface.co/BAAI/bge-reranker-base
    - LangChain Retriever: https://python.langchain.com/docs/modules/data_connection/retrievers/
    
    Args:
        embeddings_model: The embeddings model used for the vector store
        query_instruction: Instruction for query processing
        selected_chapter_id: Optional chapter ID to filter results
        
    Returns:
        ContextualCompressionRetriever: Configured retriever with re-ranking
    """
    try:
        print("Loading FAISS vector store...")
        if not VECTORSTORE_PATH.exists():
            print(f"❌ Vector store not found at {VECTORSTORE_PATH}. Please run the ingestion process first.")
            sys.exit(1)
            
        # Load the FAISS index with the embeddings model
        vectorstore = FAISS.load_local(
            str(VECTORSTORE_PATH),
            embeddings_model,
            allow_dangerous_deserialization=True  # Required for FAISS deserialization
        )

        # Configure search parameters
        search_kwargs = {"k": 10}  # Retrieve top 10 documents initially
        
        # Apply chapter filter if specified
        if selected_chapter_id and selected_chapter_id != "all":
            print(f"🔍 Filtering retrieval for chapter: {selected_chapter_id}")
            search_kwargs["filter"] = {"chapter_id": selected_chapter_id}
        else:
            print("🌐 No chapter filter applied (retrieving from all chapters).")

        # Create base retriever from FAISS index
        base_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        
        # Initialize cross-encoder for re-ranking
        # BGE-Reranker provides better semantic understanding than pure vector similarity
        print("Initializing BGE-Reranker for improved retrieval quality...")
        reranker = HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-base",  # Pre-trained re-ranking model
            model_kwargs={"max_length": 512}  # Maximum sequence length for the model
        )
        
        # Create re-ranker that will reorder the top 4 results
        compressor = CrossEncoderReranker(
            model=reranker,
            top_n=4  # Only re-rank and return top 4 most relevant results
        )
        
        # Combine the base retriever with the re-ranker
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        print("✅ Retriever with re-ranking is ready.")
        return compression_retriever
        
    except Exception as e:
        print(f"❌ Error initializing retriever: {e}")
        sys.exit(1)

def setup_qa_chain(llm, retriever):
    """Sets up the RetrievalQA chain with a custom prompt."""
    print("Setting up QA chain...")
    prompt_template = """You are an expert historian specializing in the content of the provided document. Your task is to provide accurate answers based ONLY on the following context.
CONTEXT:
{context}
QUESTION:
{question}
INSTRUCTIONS:
1. Analyze the context carefully to find the most relevant information to answer the question.
2. Formulate a comprehensive answer based exclusively on the provided text.
3. For each piece of information you use, cite the source page number, like this: [p. 12].
4. If the context does not contain the answer, state clearly: "The provided document does not contain information on this topic." Do not use outside knowledge.
ANSWER:
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": PROMPT}
    )
    print("QA chain setup complete.")
    return qa_chain

def generate_question_from_chapter_content(retriever, llm, zpd_score: float, selected_chapter_title: str, previous_questions: set = None):
    """
    Generates a unique question and its expected answer based on content retrieved
    from a specific chapter or the entire document, with difficulty adjusted by ZPD score.
    
    Args:
        retriever: The document retriever
        llm: The language model to use
        selected_chapter_title: Title of the chapter to generate questions from
        previous_questions: Set of previously asked questions to avoid repetition
        zpd_score: The user's Zone of Proximal Development score (1.0-10.0)
        
    Returns:
        tuple: (question, answer, difficulty_level)
    """
    if previous_questions is None:
        previous_questions = set()
        
    # Map ZPD score to difficulty level
    if zpd_score < 4.0:
        difficulty = "beginner"
        instruction = "Focus on basic facts, dates, and key terms."
    elif zpd_score < 7.0:
        difficulty = "intermediate"
        instruction = "Ask about causes, effects, and basic analysis."
    else:
        difficulty = "advanced"
        instruction = "Require critical thinking, comparison, and evaluation."
    
    # Define different question types to ensure variety
    question_types = [
        "a cause-and-effect question",
        "a comparison question between two events or concepts",
        "a question about historical significance",
        "a question about primary sources or evidence",
        "a question about different historical perspectives",
        "a question about long-term consequences",
        "a question about historical context"
    ]
    
    # Define different content aspects to focus on
    content_aspects = [
        "key events",
        "important figures",
        "main themes",
        "historical context",
        "primary sources",
        "causes and effects",
        "different perspectives"
    ]
    
    max_attempts = 5
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        print(f"\n🧠 Generating a {difficulty}-level question from {selected_chapter_title} (Attempt {attempt}/{max_attempts})...")
        
        try:
            # Select a random aspect to focus on
            focus_aspect = random.choice(content_aspects)
            
            # Retrieve relevant context with a specific focus
            retrieved_docs = retriever.get_relevant_documents(f"{selected_chapter_title} {focus_aspect}", k=5)
            if not retrieved_docs:
                print("No relevant documents found, trying a different approach...")
                continue
                
            # Filter documents to only include those from the selected chapter
            filtered_docs = [doc for doc in retrieved_docs if doc.metadata.get('chapter_title') == selected_chapter_title]
            if not filtered_docs:
                print(f"No documents found for chapter: {selected_chapter_title}")
                continue
                
            context = "\n\n".join([doc.page_content for doc in filtered_docs])
            
            # Select a random question type for variety
            question_type = random.choice(question_types)
            
            # Generate question and answer specific to the difficulty level
            prompt = f"""You are a history professor creating exam questions. Your task is to generate a {difficulty}-level question.

CHAPTER: {selected_chapter_title}
FOCUS ASPECT: {focus_aspect}
DIFFICULTY: {difficulty}
QUESTION TYPE: {question_type}

INSTRUCTIONS:
1. Create ONE {difficulty}-level history question based on the context.
2. {instruction}
3. The question should be clear, specific, and require understanding of the material.
4. Make sure the question is not too broad or too narrow.
5. Provide a detailed answer (2-3 sentences) that demonstrates {difficulty}-level understanding.
6. Format your response exactly as shown below:

QUESTION: [Your question here?]
ANSWER: [Your answer here.]

CONTEXT:
{context}

Now, generate the question and answer:"""
            
            response = llm.invoke([
                SystemMessage(content="""You are a history professor creating unique exam questions. 
                Ensure each question is distinct and tests different aspects of the material."""),
                HumanMessage(content=prompt)
            ]).content

            # Parse the response
            if "QUESTION:" in response and "ANSWER:" in response:
                question = response.split("QUESTION:", 1)[1].split("ANSWER:")[0].strip()
                answer = response.split("ANSWER:", 1)[1].strip()
                
                # Validate question format and check for uniqueness
                if not question.endswith('?'):
                    question = question.rstrip('.') + '?'
                
                # Check if this question is too similar to previous ones
                is_unique = True
                q_lower = question.lower()
                for prev_q in previous_questions:
                    if q_lower == prev_q.lower() or \
                       q_lower in prev_q.lower() or \
                       prev_q.lower() in q_lower or \
                       fuzz.ratio(q_lower, prev_q.lower()) > 80:  # Using fuzzy matching
                        is_unique = False
                        break
                
                # Check if the question focuses on the same aspect as previous questions
                if is_unique:
                    # Check if we've asked similar aspect questions recently
                    recent_aspects = set()
                    for prev_q in previous_questions:
                        if len(recent_aspects) >= 3:  # Allow max 3 questions of same aspect in a row
                            break
                        if fuzz.ratio(q_lower, prev_q.lower()) > 60:  # Similar question
                            recent_aspects.add(focus_aspect)
                    
                    if len(recent_aspects) < 3:  # Only allow if not too many similar aspects
                        previous_questions.add(question)
                        return question, answer, difficulty
                    else:
                        print(f"Too many similar aspect questions ({focus_aspect}), trying different aspect...")
                else:
                    print("Generated a similar question, trying again...")
                    
        except Exception as e:
            print(f"❌ Error generating question: {e}")
    
    # If we've tried max_attempts times, return a default question
    print("⚠️ Could not generate a unique question after multiple attempts. Using a default question.")
    default_questions = {
        "beginner": (f"What was a key event in {selected_chapter_title}?", 
                    "The chapter discusses significant historical events and their importance.", "beginner"),
        "intermediate": (f"What were the main causes and effects of a major event in {selected_chapter_title}?",
                       "The chapter analyzes how various factors contributed to historical developments and their consequences.", "intermediate"),
        "advanced": (f"How did different perspectives shape the outcomes in {selected_chapter_title}? Analyze the evidence.",
                    "The chapter presents multiple viewpoints and evidence that influenced historical interpretations and outcomes.", "advanced")
    }
    return default_questions[difficulty]

def generate_hint(question: str, expected_answer: str, zpd_score: float, llm) -> str:
    """Generate a hint based on the ZPD score.
    
    Args:
        question: The question being asked
        expected_answer: The expected answer
        zpd_score: The student's ZPD score (1.0-10.0)
        llm: The language model to use
        
    Returns:
        A hint string tailored to the student's ZPD level
    """
    # Determine hint style and detail level based on ZPD score
    if zpd_score < 4.0:  # Beginner
        hint_style = "simple and direct"
        detail_level = "very detailed"
        guidance = "Break down the problem into smaller steps and provide clear, specific guidance. " \
                 "Use simple language and concrete examples. The hint should be quite explicit " \
                 "but still require some thinking to connect to the answer."
    elif zpd_score < 7.0:  # Intermediate
        hint_style = "thought-provoking"
        detail_level = "moderately detailed"
        guidance = "Provide guidance that helps the student think through the problem themselves. " \
                 "Ask leading questions or point to key concepts. The hint should require some " \
                 "critical thinking to connect to the answer."
    else:  # Advanced
        hint_style = "subtle and thought-provoking"
        detail_level = "minimal"
        guidance = "Provide a subtle nudge in the right direction. The hint should be quite " \
                 "minimal and require the student to do most of the thinking. Focus on " \
                 "broader concepts rather than specific details."
    
    prompt = f"""You are a helpful history tutor providing a {hint_style} hint for a student.

QUESTION: {question}
EXPECTED ANSWER: {expected_answer}

INSTRUCTIONS:
1. Generate a {detail_level} hint that helps the student without giving away the answer.
2. {guidance}
3. The hint should be 1-2 sentences maximum.
4. Do NOT include the answer in the hint.
5. Focus on the key concept or approach needed.

HINT:"""
    
    try:
        response = llm.invoke([
            SystemMessage(content=f"You are a history tutor providing a {hint_style} hint. Your hints are {detail_level}."),
            HumanMessage(content=prompt)
        ])
        
        # Clean up the response
        hint = response.content.strip()
        if 'hint:' in hint.lower():
            hint = hint.split('hint:', 1)[1].strip()
        
        # Ensure the hint is not too revealing for the student's level
        if zpd_score < 4.0 and len(hint.split()) > 30:  # For beginners, keep hints concise
            hint = ' '.join(hint.split()[:30]) + '...'
        elif zpd_score >= 7.0 and len(hint.split()) > 15:  # For advanced, keep hints very brief
            hint = ' '.join(hint.split()[:15]) + '...'
            
        return hint
        
    except Exception as e:
        print(f"Error generating hint: {e}")
        # Fallback hints based on ZPD
        if zpd_score < 4.0:
            return "Think about the key concepts we've discussed. What's the main idea behind this question?"
        elif zpd_score < 7.0:
            return "Consider how different factors might be connected in this situation."
        else:
            return "What patterns or themes can you identify that might be relevant here?"

def analyze_student_answer(question: str, student_answer: str, expected_answer: str, llm, zpd_score: float):
    """
    Analyzes the student's answer and evaluates its correctness.
    
    Args:
        question: The question that was asked
        student_answer: The student's answer to evaluate
        expected_answer: The expected answer
        llm: The language model to use
        zpd_score: The student's ZPD score (1.0-10.0)
        
    Returns:
        Dictionary with analysis results including feedback and hint if needed
    """
    try:
        # First, check if the answer is relevant
        prompt = f"""Is this answer relevant to the question? Answer ONLY 'yes' or 'no'.
        
        Question: {question}
        Answer: {student_answer}"""
        
        response = llm.invoke([
            SystemMessage(content="You are a history professor evaluating answer relevance."),
            HumanMessage(content=prompt)
        ]).content.strip().lower()
        
        if 'no' in response:
            return {
                'is_correct': False,
                'feedback': "Your answer doesn't seem to address the question. Please focus on the specific topic being asked about.",
                'score': 0.0,
                'hint': generate_hint(question, expected_answer, zpd_score, llm)
            }
        
        # If relevant, evaluate correctness
        prompt = f"""Evaluate this answer as 'correct', 'partially correct', or 'incorrect'.
        
        Question: {question}
        Expected Answer: {expected_answer}
        Student's Answer: {student_answer}
        
        Respond with ONLY one of: correct, partially correct, incorrect"""
        
        evaluation = llm.invoke([
            SystemMessage(content="You are a history professor evaluating answer correctness."),
            HumanMessage(content=prompt)
        ]).content.strip().lower()
        
        is_correct = evaluation == 'correct'
        is_partial = 'partial' in evaluation
        score = 1.0 if is_correct else (0.5 if is_partial else 0.0)
        
        # Generate appropriate feedback
        if is_correct:
            feedback = "✅ Correct! Your answer demonstrates good understanding of the topic."
        elif is_partial:
            feedback = "⚠️ Partially correct. You're on the right track, but there's room for improvement."
        else:
            feedback = "❌ Incorrect. Let's review this concept together."
        
        # Generate hint if answer isn't fully correct
        hint = None
        if not is_correct:
            hint = generate_hint(question, expected_answer, zpd_score, llm)
        
        return {
            'is_correct': is_correct,
            'score': score,
            'feedback': feedback,
            'hint': hint
        }
            
    except Exception as e:
        print(f"Error in analyze_student_answer: {e}")
        return {
            'is_correct': False,
            'feedback': "I encountered an error evaluating your answer. Please try again.",
            'score': 0.0,
            'hint': "Consider rephrasing your answer or providing more details."
        }

def get_feedback_on_answer(user_answer: str, expected_answer: str, question: str, llm, context: str = "", zpd_score: float = 2.5):   
    """
    Evaluates the user's answer and provides feedback with ZPD-based hints.
    
    Args:
        user_answer: The student's answer
        expected_answer: The expected correct answer
        question: The question that was asked
        llm: The language model to use
        zpd_score: The student's ZPD score (1.0-10.0)
        
    Returns:
        A tuple of (feedback_message, is_correct, analysis)
    """
    try:
        # Check if expected answer is present
        if not expected_answer:
            return (
                "I don't have an expected answer for this question. Please try again.",
                False,
                {'hint': "Consider asking a different question"}
            )
        # Check if answer is too short
        if len(user_answer.split()) < 3:
            return (
                "🤔 Your answer seems quite brief. Could you elaborate more? Try to explain your thinking in more detail.",
                False,
                {'hint': generate_hint(question, expected_answer, zpd_score, llm)}
            )
        
        # Analyze the answer
        analysis = analyze_student_answer(question, user_answer, expected_answer, llm, zpd_score)
        
        # Build feedback message
        feedback_parts = [analysis['feedback']]
        
        # Add hint if available and answer isn't correct
        # if analysis.get('hint') and not analysis['is_correct']:
        #     feedback_parts.append(f"\n💡 Hint: {analysis['hint']}")
        
        # Add closing note
        if analysis['is_correct']:
            feedback_parts.append("\n👍 Great job! You've demonstrated good understanding of the topic.")
        else:
            feedback_parts.append("\n💭 Take a moment to review the material and try again. You can do it!")
        
        return (
            "\n".join(feedback_parts),
            analysis['is_correct'],
            analysis
        )
            
    except Exception as e:
        print(f"Error in get_feedback_on_answer: {e}")
        return (
            "I had trouble evaluating your response. Please try rephrasing your answer.",
            False,
            {'hint': "Consider providing more specific details in your answer."}
        )

# def ask_question(qa_chain, question):
#     """
#     This function is largely deprecated for the interactive quiz mode,
#     as the main loop directly handles question generation and feedback.
#     It remains as a utility for potential direct QA queries.
#     """
#     print("\n🔍 Processing your question with RAG chain (utility function, not used in main quiz loop)...")
#     try:
#         result = qa_chain.invoke({"query": question})
#         print(f"\n{'='*80}\n📝 RAG System's Answer (for reference/debugging):\n{result['result']}\n{'='*80}\n")
#         if result.get("source_documents"):
#             print("📚 Sources (for reference/debugging):")
#             for doc in result["source_documents"]:
#                 page_num, chapter_id, chapter_title, relevance_score = (
#                     doc.metadata.get('page', 'N/A'), doc.metadata.get('chapter_id', 'N/A'),
#                     doc.metadata.get('chapter_title', 'N/A'), doc.metadata.get('relevance_score', 'N/A')
#                 )
#                 score_str = f"{relevance_score:.4f}" if isinstance(relevance_score, float) else relevance_score
#                 print(f"  - Source: {doc.metadata.get('source', 'N/A')}, Page: {page_num}, "
#                       f"Chapter: {chapter_title} ({chapter_id}), Relevance Score: {score_str}")
#         else:
#             print("ℹ️ No sources were returned for this answer.")
#     except Exception as e:
#         print(f"An error occurred while asking the question: {e}")

# --- Main Application Logic ---
def main():
    """Main function to run the RAG system."""
    # Initialize student manager and get/create student session
    from student_manager import StudentManager, StudentSession
    
    student_mgr = StudentManager()
    try:
        # This will prompt for student ID
        session = student_mgr.get_or_create_student()
        
        # If we get here, login was successful
        print("\n=== Login Successful! ===")
        print(f"Welcome, {session.student_name}!")
        print(f"Your current ZPD score: {session.current_zpd:.1f}")
        
        
    except Exception as e:
        print(f"\nError: {str(e)}")
    
    
    current_zpd = session.current_zpd
    check_environment()
    
    try:
        chapter_map_data = load_chapter_map(CHAPTER_MAP_PATH)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    
    if not VECTORSTORE_PATH.exists():
        print("Vector store not found. Starting the data ingestion process...")
        documents = extract_text_with_metadata(PDF_PATH, chapter_map_data)
        chunks = split_documents(documents)
        create_and_save_vectorstore(chunks)
    else:
        print("Existing vector store found. Skipping ingestion.")
        
    embeddings_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True},
        query_instruction="Represent this sentence for searching relevant passages:"
    )
    
    llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.7, max_tokens=1500)

    print("\nAvailable Chapters:")
    print("0. All Chapters (Quiz Mode - questions from entire document)")
    for i, chapter in enumerate(chapter_map_data, 1):
        print(f"{i}. {chapter['title']} (ID: {chapter['id']})")
    
    selected_chapter_input = input("Enter the number of the chapter to query (e.g., 1, 2, or 0 for all): ").strip()
    
    selected_chapter_id = "all"
    selected_chapter_title = "All Chapters"
    try:
        if selected_chapter_input != "0":
            chapter_index = int(selected_chapter_input) - 1
            if 0 <= chapter_index < len(chapter_map_data):
                selected_chapter_id = chapter_map_data[chapter_index]["id"]
                selected_chapter_title = chapter_map_data[chapter_index]["title"]
            else:
                print("Invalid chapter number. Defaulting to 'All Chapters' quiz mode.")
    except ValueError:
        print("Invalid input. Defaulting to 'All Chapters' quiz mode.")

    print(f"You selected: {selected_chapter_title} for quiz mode.")

    retriever = load_retriever_and_reranker(embeddings_model, embeddings_model.query_instruction, selected_chapter_id)
    qa_chain = setup_qa_chain(llm, retriever)
    
    print("\n✅ RAG System is Ready. Type 'exit' to quit at any question prompt.")

    asked_questions_history = set()
    max_retries_for_unique_question = 5

    while True:
        try:
            generated_question, expected_answer, display_context_name = None, None, None
            
            # Debug: Show current ZPD and difficulty level
            difficulty = "beginner" if current_zpd < 4.0 else "intermediate" if current_zpd < 7.5 else "advanced"
            print(f"\n[DEBUG] Current ZPD: {current_zpd:.1f} (Level: {difficulty})")
            
            for attempt in range(max_retries_for_unique_question):
                temp_question, temp_answer, temp_display_name = generate_question_from_chapter_content(
                    retriever=retriever, 
                    llm=llm, 
                    selected_chapter_title=selected_chapter_title,
                    previous_questions=asked_questions_history, 
                    zpd_score=current_zpd  # Use the current ZPD score
                )
                
                if temp_question and temp_answer and temp_question != "Could not generate a question.":
                    normalized_question = ' '.join(temp_question.lower().split())
                    if normalized_question not in asked_questions_history:
                        generated_question, expected_answer, display_context_name = temp_question, temp_answer, temp_display_name
                        asked_questions_history.add(normalized_question)
                        break
                    else:
                        print(f"--- Duplicate question generated (attempt {attempt + 1}/{max_retries_for_unique_question}), trying again...")
                else:
                    break
            
            if generated_question and expected_answer and generated_question != "Could not generate a question.":
                print(f"\nHere's a question for you {display_context_name}:")
                print(f"Question: {generated_question}")
                print(f"Answer: {expected_answer}")
                user_answer = input("Your answer (or 'exit'): ").strip()

                if user_answer.lower() == 'exit':
                    break

                if user_answer:
                    retrieved_docs = retriever.get_relevant_documents(generated_question)
                    context = "\n".join([doc.page_content for doc in retrieved_docs[:2]])
                    
                    # Get feedback on the answer
                    feedback, is_correct, analysis = get_feedback_on_answer(
                        user_answer=user_answer, 
                        expected_answer=expected_answer,
                        question=generated_question, 
                        llm=llm, 
                        context=context,
                        zpd_score=current_zpd  # Pass current ZPD for hint generation
                    )
                    
                    # Update ZPD based on performance using ZPDCalculator
                    old_zpd = current_zpd
                    
                    # Record performance (1.0 = correct, 0.5 = partial, 0.0 = incorrect)
                    performance_score = 1.0 if is_correct else (0.5 if analysis.get('partially_correct', False) else 0.0)
                    
                    # Update ZPD using the student manager
                    old_zpd, current_zpd = student_mgr.update_student_zpd(
                        student_session=session,
                        is_correct=is_correct,
                        is_partial=analysis.get('partially_correct', False)
                    )
                    
                    # Get direction of change for debug message
                    change = current_zpd - old_zpd
                    change_str = f"+{change:.2f}" if change >= 0 else f"{change:.2f}"
                    
                    print(f"\n[DEBUG] ZPD: {old_zpd:.1f} -> {current_zpd:.1f} ({change_str})")
                    if is_correct:
                        print("Great job! Your ZPD score has increased.")
                    elif analysis.get('partially_correct', False):
                        print("Good effort! Your ZPD score has increased slightly.")
                    else:
                        print("Keep trying! Your ZPD score has been adjusted based on your performance.")
                    
                    print(f"Your new ZPD score: {current_zpd:.1f}")
                    
                    print("\n---")
                    print(f"Your answer: {user_answer}")
                    print(f"\n{feedback}")
                    
                    if not is_correct: # For both partially_correct and wrong answers
                        if input("\nWould you like a hint? (yes/no): ").lower().strip() == 'yes':
                            print(f"\n💡 Hint: {analysis['hint']}")
                        
                        if input("\nWould you like to see the full answer? (yes/no): ").lower().strip() == 'yes':
                            print(f"\nThe full correct answer is: {expected_answer}")
                else:
                    print("You didn't provide an answer.")
                
                if input("\nDo you want another question? (yes/no): ").lower().strip() != 'yes':
                    break
            else:
                try:
                    # Create a simple prompt that's easy to parse
                    simple_prompt = f"""Create one history question and its answer based on this context.
                    
                    CONTEXT:
                    {context}
                    
                    Format your response as:
                    Question: [your question here]
                    Answer: [your answer here]"""
                    
                    # Get the response
                    response = llm.invoke([
                        SystemMessage(content="You are a helpful history tutor."),
                        HumanMessage(content=simple_prompt)
                    ]).content
                    
                    # Simple parsing
                    question = re.search(r"Question: ?(.*?)(?=\n|$)", response, re.IGNORECASE)
                    answer = re.search(r"Answer: ?(.*?)(?=\n|$)", response, re.IGNORECASE)
                    
                    # If parsing failed, try to extract question and answer from the response
                    if not question or not answer:
                        lines = [line.strip() for line in response.split('\n') if line.strip()]
                        if len(lines) >= 2:
                            question = lines[0] if '?' in lines[0] else lines[1] if len(lines) > 1 else None
                            answer = lines[1] if '?' not in lines[0] and len(lines) > 1 else lines[2] if len(lines) > 2 else None
                    else:
                        question = question.group(1).strip()
                        answer = answer.group(1).strip()
                    
                    # Ensure we have valid question and answer
                    if not question or not answer:
                        question = f"What is a key event or concept from {selected_chapter_title}?"
                        answer = "The chapter covers important historical events and concepts from this period."
                    
                    # Add to previous questions to avoid repetition
                    if asked_questions_history is not None:
                        asked_questions_history.add(question)
                    
                    print(f"Question generated successfully.")
                    generated_question, expected_answer, display_context_name = question, answer, selected_chapter_title
                except Exception as e:
                    print(f"Error generating question: {e}")
                    # Return a simple default question if anything goes wrong
                    default_q = f"What is one important aspect of {selected_chapter_title}?"
                    default_a = "The chapter discusses significant historical events and their impacts."
                    generated_question, expected_answer, display_context_name = default_q, default_a, selected_chapter_title
                
                if generated_question and expected_answer and generated_question != "Could not generate a question.":
                    print(f"\nHere's a question for you {display_context_name}:")
                    print(f"Question: {generated_question}")
                    user_answer = input("Your answer (or 'exit'): ").strip()

                    if user_answer.lower() == 'exit':
                        break

                    if user_answer:
                        retrieved_docs = retriever.get_relevant_documents(generated_question)
                        context = "\n".join([doc.page_content for doc in retrieved_docs[:2]])
                        
                        feedback, is_correct, analysis = get_feedback_on_answer(
                            user_answer=user_answer, expected_answer=expected_answer,
                            question=generated_question, llm=llm, context=context
                        )
                        
                        print("\n---")
                        print(f"Your answer: {user_answer}")
                        print(f"\n{feedback}")
                        
                        if not is_correct: # For both partially_correct and wrong answers
                            if input("\nWould you like a hint? (yes/no): ").lower().strip() == 'yes':
                                print(f"\n💡 Hint: {analysis['hint']}")
                            
                            if input("\nWould you like to see the full answer? (yes/no): ").lower().strip() == 'yes':
                                print(f"\nThe full correct answer is: {expected_answer}")
                    else:
                        print("You didn't provide an answer.")
                    
                    if input("\nDo you want another question? (yes/no): ").lower().strip() != 'yes':
                        break

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
    # Initialize the student manager
