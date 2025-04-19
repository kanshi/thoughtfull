import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Milvus Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "192.168.3.11")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# Vector dimensions for the sentence transformer model
VECTOR_DIMENSION = 384  # Actual dimension from the model

# Transformer model for embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Collection names in Milvus
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_store")
CONVERSATION_COLLECTION = os.getenv("CONVERSATION_COLLECTION", "conversation_history")

# File upload settings
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
ALLOWED_EXTENSIONS = {"pdf", "txt"}

# Maximum file size (10 MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Chunk size for document processing (in characters)
CHUNK_SIZE = 1000

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# LLM Models - default configurations for CPU and GPU models
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "llama3")  # Default model to use
CPU_LLM_MODEL = os.getenv("CPU_LLM_MODEL", "mistral")  # Lightweight model for CPU
GPU_LLM_MODEL = os.getenv("GPU_LLM_MODEL", "llama3")  # Full-featured model for GPU

# Chat history settings
MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", "10"))  # Maximum messages to keep in context
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))  # Maximum document chunks to include
