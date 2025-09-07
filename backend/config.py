
import os
from dotenv import load_dotenv
load_dotenv()

# Embedding backend: 'local' (sentence-transformers) or 'openai'
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Vector DB
VECTOR_PERSIST_DIR = os.getenv("VECTOR_PERSIST_DIR", "../data/faiss_index")

# Ingest / chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))

# Guardrail thresholds
MIN_CONTEXT_TOKENS = int(os.getenv("MIN_CONTEXT_TOKENS", "50"))
