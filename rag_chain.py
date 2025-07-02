### app.py
import streamlit as st
import os
import shutil
from config import TEMP_UPLOAD_DIR
from loader import load_documents, build_vectorstore
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from config import VECTORSTORE_DIR, EMBED_MODEL, OLLAMA_MODEL  # <-- import OLLAMA_MODEL
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

def get_rag_chain(vectordb):
    llm = Ollama(model=OLLAMA_MODEL)  # <-- use OLLAMA_MODEL here
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

st.set_page_config(page_title="Universal RAG Chat", layout="centered")
st.title("📁 Local RAG System for All Docs")

# File upload section
uploaded_files = st.file_uploader("Upload your documents (PDF, DOCX, CSV, TXT):", accept_multiple_files=True)

if uploaded_files:
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    for uploaded_file in uploaded_files:
        with open(os.path.join(TEMP_UPLOAD_DIR, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

    with st.spinner("Indexing uploaded documents..."):
        docs = load_documents(TEMP_UPLOAD_DIR)
        vectordb = build_vectorstore(docs)
        st.session_state.rag_chain = get_rag_chain(vectordb)
    shutil.rmtree(TEMP_UPLOAD_DIR)
    st.success("Documents uploaded and indexed successfully!")

# Fallback: load existing vectorstore if available
if "rag_chain" not in st.session_state and os.path.exists(VECTORSTORE_DIR):
    with st.spinner("Loading existing knowledge base..."):
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        vectordb = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)
        st.session_state.rag_chain = get_rag_chain(vectordb)

# Question input
query = st.text_input(
    "Ask a question based on your documents:",
    "Summarize the key takeaways.",
    key="user_query_input"  # <-- Add a unique key here
)
if st.button("Submit", key="submit_button") and query and "rag_chain" in st.session_state:
    with st.spinner("Generating answer..."):
        response = st.session_state.rag_chain.invoke(query)
        st.markdown("### 🧠 Answer")
        st.write(response["result"])
        # Optional: Show sources
        with st.expander("Show source documents"):
            for i, doc in enumerate(response.get("source_documents", []), 1):
                st.markdown(f"**Source {i}:** {getattr(doc, 'page_content', str(doc))}")

# Multi-document summarization
if st.button("Summarize All Documents", key="summarize_button") and "rag_chain" in st.session_state:
    with st.spinner("Summarizing all content..."):
        summary_query = "Summarize all key findings across the documents."
        response = st.session_state.rag_chain.invoke(summary_query)
        st.markdown("### 🧾 Summary of All Documents")
        st.write(response["result"])
        with st.expander("Show source documents"):
            for i, doc in enumerate(response.get("source_documents", []), 1):
                st.markdown(f"**Source {i}:** {getattr(doc, 'page_content', str(doc))}")



