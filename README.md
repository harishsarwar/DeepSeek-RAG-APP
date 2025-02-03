# PDF-Based QA Chatbot with DeepSeeek


This project enables users to ask **context-aware** questions based on a **PDF document** using **FAISS vector search** and **LangChain with Ollama LLM**.

## 🚀 Features
✅ **PDF Ingestion** – Extracts text and splits it into meaningful chunks  
✅ **FAISS Vector Store** – Stores embeddings for fast retrieval  
✅ **Contextual Q&A** – Provides answers based on document content  
✅ **Streamlit UI** – Interactive chatbot interface  
✅ **Ollama LLM Integration** – Uses local LLM for response generation  

## 🔥 Use Cases
🔹 **Resume-Based Q&A** – Ask questions about a resume document  
🔹 **Legal Document Search** – Retrieve relevant legal clauses  
🔹 **Research Paper Assistant** – Extract insights from academic PDFs  
🔹 **Medical Report Analysis** – Answer queries based on medical documents  

🔹 **And show on according to the PDf**

## 📁 Project Structure
📂 project-folder
 ├── main.py                # Streamlit chatbot UI
 ├── data_ingestion.py       # PDF processing & FAISS database creation
 ├── vectorstore/            # FAISS vector storage
 ├── Resume (5).pdf          # Sample PDF file
 ├── .env                    # API keys (not included in repo)
 ├── requirements.txt        # Dependencies
 ├── README.md               # Project documentation


## ❗ Troubleshooting
If FAISS does not load data in main.py, rerun data_ingestion.py
Ensure Ollama is running at http://localhost:11434
Check if .env is correctly set





