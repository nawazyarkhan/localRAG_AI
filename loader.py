### loader.py
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from config import EMBED_MODEL, VECTORSTORE_DIR

import os


def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            documents.extend(PyPDFLoader(filepath).load())
        elif filename.endswith(".csv"):
            documents.extend(CSVLoader(filepath).load())
        elif filename.endswith(".txt"):
            documents.extend(TextLoader(filepath).load())
        elif filename.endswith(".docx"):
            documents.extend(UnstructuredWordDocumentLoader(filepath).load())
    return documents


def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=VECTORSTORE_DIR)
    vectordb.persist()
    return vectordb


def load_or_build_vectorstore(folder_path):
    if os.path.exists(VECTORSTORE_DIR):
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        return Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)
    docs = load_documents(folder_path)
    return build_vectorstore(docs)