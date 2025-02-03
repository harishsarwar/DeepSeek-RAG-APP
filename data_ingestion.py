from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")

os.environ["HUGGINGFACE_API_KEY"]=api_key 



data_path = "Resume (5).pdf"
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = "sentence-transformers/all-mpnet-base-v2" # Embeddig model from hugging face we can used random embedding model



# creating vecto database.
def create_vector_db():
    loader = PyPDFLoader(data_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    print(f"Total Chunks Created: {len(chunks)}")  # Debugging

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2", 
        model_kwargs={"token": api_key}
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"FAISS Database Created at {DB_FAISS_PATH}")



if __name__ == '__main__':
    create_vector_db()
