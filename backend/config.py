import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
DATA_DIRS = {
    "schemes": os.path.join(ROOT_DIR, "data", "core_website_data")
}

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8001))

EMBEDDING_MODEL = "thenlper/gte-small"
LLM_MODEL = "gemma4:e4b"
COLLECTION_NAME = "kerala_startup_mission"

# Single context-window size shared by ALL Ollama calls. Ollama reallocates
# (reloads) the model whenever num_ctx changes between requests, so mixing
# values across the helper calls and the main answer call defeats
# OLLAMA_KEEP_ALIVE and forces a ~10s reload every request. Keeping one value
# lets a single model instance stay resident across all calls.
LLM_NUM_CTX = 8192

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 250