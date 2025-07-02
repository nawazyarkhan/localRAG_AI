# Local LLM to help answer user questions from provided text/PDF/doc files

1.  ** Features ** 
  -   **Text Analysis**: The model can analyze text from various sources, including PDFs, Word Docs, CSVs, plain text files
  -   
Supports PDFs, Word Docs, CSVs, and plain text files.

Automatically builds and saves a ChromaDB vector store and leveraging Langchain features too. 

Streamlit-based UI for interactive document Q&A.

2. ** How to run ** 
   run using streamlit run app.py after creating the folder structure as given below (step#3)

3. . ** Folder Structure **
### 📁 Folder Structure
# .
# ├── app.py
# ├── config.py
# ├── loader.py
# ├── rag_chain.py
# ├── documents/                <-- Optional static source
# ├── rag_vectorstore/         <-- Auto-generated Chroma vector DB
# └── temp_uploads/            <-- Temporary folder for uploaded files

