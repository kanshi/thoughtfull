FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs

# Expose the port the app runs on
EXPOSE 8000

# Environment variables - these can be overridden at runtime
ENV MILVUS_HOST=host.docker.internal \
    MILVUS_PORT=19530 \
    COLLECTION_NAME=document_store \
    EMBEDDING_MODEL=all-MiniLM-L6-v2 \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    DEFAULT_LLM_MODEL=llama3 \
    CPU_LLM_MODEL=mistral \
    GPU_LLM_MODEL=llama3 \
    MAX_HISTORY_LENGTH=10 \
    MAX_CONTEXT_CHUNKS=5

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
