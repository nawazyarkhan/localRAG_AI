### app.py
import streamlit as st
from loader import load_or_build_vectorstore
from rag_chain import get_rag_chain
from config import OLLAMA_MODEL

st.set_page_config(page_title="Universal RAG Chat", layout="centered")
st.title("📁 Local RAG System for All Docs")

folder = st.text_input("Folder path containing documents (PDF, DOCX, CSV, TXT):", "./documents")

if "rag_chain" not in st.session_state:
    with st.spinner("Loading and indexing documents..."):
        vectordb = load_or_build_vectorstore(folder)
        st.session_state.rag_chain = get_rag_chain(vectordb)

query = st.text_input("Ask a question based on your documents:", "Summarize the key takeaways.")
if st.button("Submit") and query:
    with st.spinner("Generating answer..."):
        result = st.session_state.rag_chain.invoke(query)
        st.markdown("### 🧠 Answer")
        st.write(result["result"])
        # Optional: Show sources
        with st.expander("Show source documents"):
            for i, doc in enumerate(result.get("source_documents", []), 1):
                st.markdown(f"**Source {i}:** {getattr(doc, 'page_content', str(doc))}")

def get_rag_chain(vectordb):
    llm = Ollama(model=OLLAMA_MODEL)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )
    return qa_chain