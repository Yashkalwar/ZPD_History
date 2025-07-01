import os
import sys
import re
import json
from pathlib import Path
import fitz  # PyMuPDF
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "raw" / "history.pdf"
VECTORSTORE_PATH = BASE_DIR / "data" / "faiss_index_optimized"
CHAPTER_MAP_PATH = BASE_DIR / "data" / "raw" / "chapter_map.json"

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

def generate_question_from_chapter_content(retriever, llm, selected_chapter_title: str):
    """
    Generates a question and its expected answer based on content retrieved
    from a specific chapter or the entire document, focusing on historical facts/events.
    """
    display_context_name = f"from {selected_chapter_title}" if selected_chapter_title != "All Chapters" else "from the document"
    retrieval_query_base = "key historical events, figures, causes, or effects relevant to a short answer question"
    retrieval_query = f"{retrieval_query_base} from '{selected_chapter_title}'" if selected_chapter_title != "All Chapters" else retrieval_query_base
    context_description = f"Context {display_context_name}:"

    print(f"\n🧠 Generating a question {display_context_name}...")
    try:
        retrieved_docs = retriever.get_relevant_documents(retrieval_query)
        if not retrieved_docs:
            print("No relevant documents found for question generation.")
            return None, None, None

        context = "\n".join([doc.page_content for doc in retrieved_docs[:5]])

        generation_prompt_template = """You are a history quiz master creating exam-style questions.
Based *strictly and specifically* on the following historical text, generate a single, concise, open-ended question.
The question *must be directly and fully answerable from the provided context*.
Focus on a central theme, a significant event, a key cause, a specific effect, or an important figure explicitly discussed in the provided text.
The question should be concise and designed to elicit a factual, one-line answer, typical of a low-mark exam question.
Avoid general overview questions if the context provides specific details.
Do NOT add phrases like "in this chapter" or "from the text" at the end of the question.

Also, provide the direct and factual answer to that question based *only* on the provided context. The answer should also be concise and suitable for a one-line response.

{context_description}
{context}

Format your output as follows:
Question: <Your generated historical question here>
Answer: <The direct historical answer to your question here>
"""
        
        chain = PromptTemplate(template=generation_prompt_template, input_variables=["context", "context_description"]) | llm 
        llm_response = chain.invoke({"context": context, "context_description": context_description})
        
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

def get_feedback_on_answer(user_answer: str, expected_answer: str, question: str, llm):
    """Compares the user's answer to the expected answer and provides concise feedback."""
    print("\n🧐 Analyzing your answer...")
    feedback_prompt_template = """You are an AI tutor providing feedback on a student's answer to a history question.
Here is the original question asked: "{question}"
Compare the user's answer to the expected correct answer. Your evaluation should be based on whether the user's answer provides the core factual information requested by the question.
Crucially:
- If the original question already specifies details like location, year, or context (e.g., "in 1936", "in Spain", "that challenged the Treaty of Versailles"), the user's answer DOES NOT need to explicitly repeat those details to be considered correct. Focus on the core event, person, or concept being asked about.
- DO NOT introduce any external information, details, or topics not directly related to THIS SPECIFIC question, user answer, and expected answer.
- Ensure your feedback is directly relevant to the user's answer in relation to the expected answer, considering the information already present in the question.
Provide a one-line feedback response.
Categorize the feedback as:
- 'Correct' if the user's answer fully captures the essential information required to answer the question, considering the context already provided in the question.
- 'Partially Correct' if the user's answer has some correct elements but misses key factual information or contains minor inaccuracies based on the expected answer.
- 'Incorrect' if the user's answer is largely wrong or completely misses the point.
If 'Partially Correct' or 'Incorrect', briefly state the main reason or what was missed, focusing on conciseness. Do not reveal the full correct answer.
Expected Answer: {expected_answer}
User's Answer: {user_answer}
Feedback:"""

    feedback_chain = PromptTemplate(template=feedback_prompt_template, input_variables=["expected_answer", "user_answer", "question"]) | llm
    try:
        feedback_text = feedback_chain.invoke({"expected_answer": expected_answer, "user_answer": user_answer, "question": question}).content.strip()
        print(f"Feedback: {feedback_text}") 
        return feedback_text
    except Exception as e:
        print(f"Error getting feedback: {e}")
        return "Could not generate feedback."

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
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, max_tokens=1500) # Reusing this LLM instance

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
    qa_chain = setup_qa_chain(llm, retriever) # qa_chain setup remains, but not directly used by quiz generation/feedback
    
    print("\n✅ RAG System is Ready. Type 'exit' to quit at any question prompt.")

    asked_questions_history = set()
    max_retries_for_unique_question = 5

    while True:
        try:
            generated_question, expected_answer, display_context_name = None, None, None
            
            for attempt in range(max_retries_for_unique_question):
                temp_question, temp_answer, temp_display_name = generate_question_from_chapter_content(
                    retriever, llm, selected_chapter_title
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
                print(f"\nHere's a question for you {display_context_name}:\nQuestion: {generated_question}")
                user_answer = input("Your answer (or 'exit'): ").strip()

                if user_answer.lower() == 'exit':
                    break

                if user_answer:
                    print(f"\n--- Expected Answer (for reference): {expected_answer}") 
                    get_feedback_on_answer(user_answer, expected_answer, generated_question, llm)
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