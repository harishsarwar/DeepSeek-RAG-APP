import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
import os
from dotenv import load_dotenv

load_dotenv()

# Environment setup
repo_id = "deepseek-r1:1.5b"  # Use the selected DeepSeek model
api_key = os.getenv("HUGGINGFACE_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = api_key

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = "sentence-transformers/all-mpnet-base-v2"  # Embedding model for chunk embeddings

# Load the vector database
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"token": api_key})

    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"FAISS database not found at {DB_FAISS_PATH}. Run data_ingestion.py first.")

    try:
        faiss_db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        print(f"FAISS Database Loaded Successfully. Number of Documents: {len(faiss_db.docstore._dict)}")
        return faiss_db
    except Exception as e:
        raise RuntimeError(f"Error loading FAISS database: {str(e)}")

# Initialize the Ollama chat engine
llm_engine = ChatOllama(
    model=repo_id,
    base_url="http://localhost:11434",  # Replace with the appropriate URL for your Ollama service
    temperature=0.3
)

# System message for AI behavior
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI assistant. Provide precise, correct responses only if data available in {DB_FAISS_PATH}. Else return nothing"
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hello! Ask me anything Or through the PDF."}]

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input and processing
user_query = st.chat_input("Ask any question...")

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain(docs):
    prompt_sequence = [system_prompt]
    for doc in docs:
        prompt_sequence.append(AIMessagePromptTemplate.from_template(doc.page_content))
    prompt_sequence.append(HumanMessagePromptTemplate.from_template(user_query))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    db = load_vector_db()
    docs = db.similarity_search(user_query, k=2)  # Try increasing `k` for more results

    print(f"Query: {user_query} | Found {len(docs)} results")  # Debugging

    if docs:
        st.session_state.message_log.append({"role": "user", "content": user_query})

        with st.spinner("Processing..."):
            prompt_chain = build_prompt_chain(docs)
            ai_response = generate_ai_response(prompt_chain)

        st.session_state.message_log.append({"role": "ai", "content": ai_response})
        st.rerun()
    else:
        st.warning("No relevant information found in the database.")
