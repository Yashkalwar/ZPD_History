import os
import sys
import re
import json
from pathlib import Path
import fitz  # PyMuPDF
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# --- Constants ---
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "raw" / "history.pdf"
VECTORSTORE_PATH = BASE_DIR / "data" / "faiss_index_optimized"
CHAPTER_MAP_PATH = BASE_DIR / "data" / "raw" / "chapter_map.json"

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
    """Creates and saves a FAISS vector store from document chunks."""
    try:
        print("Initializing BGE embeddings for vector store creation...")
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-base-en-v1.5", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True}
        )
        print(f"Creating FAISS index from {len(chunks)} chunks...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(VECTORSTORE_PATH))
        print(f"Vector store created and saved successfully at: {VECTORSTORE_PATH}")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        sys.exit(1)

def load_retriever_and_reranker(embeddings_model, query_instruction: str, selected_chapter_id: str = None):
    """Loads the vector store and sets up a sophisticated retriever with a re-ranking stage."""
    try:
        print("Loading vector store...")
        if not VECTORSTORE_PATH.exists():
             print(f"Vector store not found at {VECTORSTORE_PATH}. Please run the ingestion process first.")
             sys.exit(1)
        vectorstore = FAISS.load_local(str(VECTORSTORE_PATH), embeddings_model, allow_dangerous_deserialization=True)

        search_kwargs = {"k": 10}
        if selected_chapter_id and selected_chapter_id != "all":
            print(f"Filtering retrieval for chapter: {selected_chapter_id}")
            search_kwargs["filter"] = {"chapter_id": selected_chapter_id}
        else:
            print("No chapter filter applied (retrieving from all chapters).")

        base_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        print("Initializing Cross-Encoder for re-ranking...")
        compressor = CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base"), top_n=4)
        print("Retriever with re-ranking is ready.")
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
    except Exception as e:
        print(f"Error loading retriever: {e}")
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

def generate_question_from_chapter_content(retriever, llm, selected_chapter_title: str, previous_questions: set = None):
    """
    Generates a unique question and its expected answer based on content retrieved
    from a specific chapter or the entire document, focusing on historical facts/events.
    """
    if previous_questions is None:
        previous_questions = set()
        
    display_context_name = f"from {selected_chapter_title}" if selected_chapter_title != "All Chapters" else "from the document"
    
    query_variations = [
        "specific historical events with exact dates", "key political figures and their roles",
        "important treaties, agreements, or alliances", "major military conflicts or battles",
        "significant social or economic developments", "cultural or technological advancements",
        "diplomatic relations between countries", "causes and consequences of major events",
        "quotes from important historical figures", "changes in political systems or governments"
    ]
    if selected_chapter_title != "All Chapters":
        query_variations = [f"{q} in {selected_chapter_title}" for q in query_variations]
    
    import random
    retrieval_query = f"{random.choice(query_variations)} {random.choice(['focusing on different aspects', 'with specific details', 'that\'s not commonly known', 'that tests deeper understanding', 'with precise historical context'])} that would make a good quiz question"
    context_description = f"Context {display_context_name}:"

    print(f"\n🧠 Generating a question {display_context_name}...")
    try:
        retrieved_docs = retriever.get_relevant_documents(retrieval_query, k=10)
        if not retrieved_docs:
            print("No relevant documents found for question generation.")
            return None, None, None
            
        random.shuffle(retrieved_docs)
        context = "\n---\n".join([doc.page_content for doc in retrieved_docs[:3]])

        generation_prompt_template = """You are a history quiz master creating unique exam-style questions with strict one-line answers.
Based *strictly and specifically* on the following historical text, generate a single, concise question.
The question MUST be answerable in exactly one line and directly from the provided context.
Focus on specific facts, dates, names, or events that have definitive, concise answers.
The answer must be a single fact or phrase that fits in one line (max 15 words).
Do NOT generate questions that require explanations, lists, or multiple sentences as answers.
Do NOT add phrases like "in this chapter" or "from the text" at the end of the question.
Make sure the question is different from these previous questions: {previous_questions}
Generate a completely new question and its one-line answer based *only* on the provided context.

{context_description}
{context}

Format your output as follows:
Question: <Your generated historical question here>
Answer: <The direct historical answer to your question here>
"""
        
        temp_llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = generation_prompt_template.format(
            context=context,
            context_description=context_description,
            previous_questions='\n- '.join(previous_questions) if previous_questions else 'No previous questions'
        )
        
        llm_response = temp_llm.invoke([
            SystemMessage(content="You are a history quiz master creating unique questions."),
            HumanMessage(content=prompt)
        ])
        
        response_text = llm_response.content
        question_match = re.search(r"Question: (.*)", response_text, re.DOTALL)
        answer_match = re.search(r"Answer: (.*)", response_text, re.DOTALL)

        generated_question = question_match.group(1).strip() if question_match else "Could not generate a question."
        expected_answer = answer_match.group(1).strip() if answer_match else "Could not generate an answer."

        if generated_question == "Could not generate a question." or expected_answer == "Could not generate an answer.":
            print(f"LLM did not parse correctly. Raw response:\n{response_text}")
        else:
            print("Question Generated Successfully.")
        return generated_question, expected_answer, display_context_name

    except Exception as e:
        print(f"Error generating question: {e}")
        return None, None, None

def analyze_student_answer(question: str, context: str, student_answer: str, expected_answer: str, llm) -> dict:
    """
    Analyzes the student's answer and categorizes it as correct, partially correct, or wrong.
    Returns a dictionary with analysis results.
    """
    try:
        analysis_prompt = """You are a helpful history tutor evaluating a student's answer. Analyze the following:
        
        Question: {question}
        Context: {context}
        Expected Answer: {expected_answer}
        Student's Answer: {student_answer}
        
        Your task is to:
        1. Be encouraging and focus on the learning process.
        2. If the answer is close but not exactly right, consider it 'partially_correct'.
        3. For 'partially_correct' answers:
           - 'explanation' should only state "Your answer is partially correct."
           - 'suggestion' should guide them to think differently without revealing the answer directly.
              - Focus on the type of information they're missing (e.g., "Think about other countries involved")
              - Ask guiding questions (e.g., "What other regions were part of this plan?")
              - Point to general themes or categories (e.g., "Consider both western and eastern expansion")
        4. For 'wrong' answers:
           - 'explanation' should only state "Your answer is incorrect."
           - 'suggestion' should provide a very general nudge (e.g., "Review the key events of this period")
        5. Never repeat the expected answer or key terms from it in the suggestion.

        Return your analysis in this exact JSON format:
        {
            "verdict": "correct|partially_correct|wrong",
            "explanation": "A brief response based on the rules above.",
            "suggestion": "A subtle hint that guides without giving away the answer."
        }
        
        Remember: The goal is to make them think, not to tell them the answer.
        """
        
        messages = [
            SystemMessage(content=analysis_prompt),
            HumanMessage(content=json.dumps({
                "question": question, "context": context,
                "expected_answer": expected_answer, "student_answer": student_answer
            }))
        ]
        
        response = llm.invoke(messages, temperature=0.7)
        
        try:
            analysis = json.loads(response.content.strip()) if hasattr(response, 'content') else json.loads(response)
            return analysis
        except json.JSONDecodeError as je:
            print(f"JSON decode error: {je}\nRaw response: {response.content if hasattr(response, 'content') else response}")
            raise
        
    except Exception as e:
        print(f"Error analyzing answer: {e}\nType: {type(e).__name__}\nArgs: {e.args}")
        return {
            "verdict": "wrong", # Default to wrong if an error occurs
            "explanation": "Your answer could not be fully analyzed.",
            "suggestion": "Consider reviewing the topic for key details."
        }

def get_feedback_on_answer(user_answer: str, expected_answer: str, question: str, llm, context: str = ""):
    """
    Compares the user's answer to the expected answer and provides detailed feedback.
    Returns a tuple of (feedback_message, is_correct, analysis)
    """
    if not user_answer.strip():
        return "Please provide an answer.", False, None
        
    analysis = analyze_student_answer(question, context, user_answer, expected_answer, llm)
    
    if analysis["verdict"] == "correct":
        return f"✅ Correct! {analysis['explanation']}", True, analysis
    elif analysis["verdict"] == "partially_correct":
        return f"⚠️ Partially Correct. {analysis['explanation']}", False, analysis
    else:  # wrong
        return f"❌ Incorrect. {analysis['explanation']}", False, analysis

def ask_question(qa_chain, question):
    """
    This function is largely deprecated for the interactive quiz mode,
    as the main loop directly handles question generation and feedback.
    It remains as a utility for potential direct QA queries.
    """
    print("\n🔍 Processing your question with RAG chain (utility function, not used in main quiz loop)...")
    try:
        result = qa_chain.invoke({"query": question})
        print(f"\n{'='*80}\n📝 RAG System's Answer (for reference/debugging):\n{result['result']}\n{'='*80}\n")
        if result.get("source_documents"):
            print("📚 Sources (for reference/debugging):")
            for doc in result["source_documents"]:
                page_num, chapter_id, chapter_title, relevance_score = (
                    doc.metadata.get('page', 'N/A'), doc.metadata.get('chapter_id', 'N/A'),
                    doc.metadata.get('chapter_title', 'N/A'), doc.metadata.get('relevance_score', 'N/A')
                )
                score_str = f"{relevance_score:.4f}" if isinstance(relevance_score, float) else relevance_score
                print(f"  - Source: {doc.metadata.get('source', 'N/A')}, Page: {page_num}, "
                      f"Chapter: {chapter_title} ({chapter_id}), Relevance Score: {score_str}")
        else:
            print("ℹ️ No sources were returned for this answer.")
    except Exception as e:
        print(f"An error occurred while asking the question: {e}")

# --- Main Application Logic ---
def main():
    """Main function to run the RAG system."""
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
            
            for attempt in range(max_retries_for_unique_question):
                temp_question, temp_answer, temp_display_name = generate_question_from_chapter_content(
                    retriever=retriever, llm=llm, selected_chapter_title=selected_chapter_title, previous_questions=asked_questions_history
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
                            print(f"\n💡 Hint: {analysis['suggestion']}")
                        
                        if input("\nWould you like to see the full answer? (yes/no): ").lower().strip() == 'yes':
                            print(f"\nThe full correct answer is: {expected_answer}")
                else:
                    print("You didn't provide an answer.")
                
                if input("\nDo you want another question? (yes/no): ").lower().strip() != 'yes':
                    break
            else:
                print("Could not generate a relevant question after multiple attempts. This might happen if the selected content is too sparse for specific questions.")
                if input("Try generating another question? (yes/no): ").lower().strip() != 'yes':
                    break

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()