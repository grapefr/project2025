
import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from .config import VECTOR_PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_BACKEND, OPENAI_API_KEY

def get_embeddings():
    if EMBEDDING_BACKEND == "openai" and OPENAI_API_KEY:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    else:
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        return SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_documents(root_dir: str):
    root = Path(root_dir)
    docs = []
    for p in root.rglob('*'):
        if p.is_file():
            if p.suffix.lower() in ['.txt', '.md']:
                docs.extend(TextLoader(str(p), encoding='utf-8').load())
            elif p.suffix.lower() == '.pdf':
                docs.extend(PyPDFLoader(str(p)).load())
    return docs

def build_vectorstore(raw_dir: str = "../data/raw_texts", persist_dir: str = VECTOR_PERSIST_DIR):
    os.makedirs(persist_dir, exist_ok=True)
    docs = load_documents(raw_dir)
    if not docs:
        print("No documents found in", raw_dir)
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(persist_dir)
    print(f"Saved vectorstore to {persist_dir}, chunks: {len(chunks)}")

if __name__ == '__main__':
    build_vectorstore()
