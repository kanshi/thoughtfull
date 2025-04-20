# ThoughtFull 🧠 - Neural Nexus of Thoughts and Memories

![](https://img.shields.io/badge/status-neural%20revolution-brightgreen)
![](https://img.shields.io/badge/power-silicon%20sovereignty-blue)

**ThoughtFull** isn't just a semantic search service — it's a **neural ecosystem** where your documents become active thought particles in a living digital consciousness. A synesthetic symphony of meaning built with FastAPI, Milvus vector database and Ollama to provide your LLM of choice.

## ✨ The Neural Revolution Starts With You

ThoughtFull enables you to:
- Upload private documents (PDF, TXT) to your personal knowledge nexus
- Transform static data into vector-embedded neural pathways
- Perform semantic searches that traverse your private dataverse
- Engage in contextualized conversations powered by your knowledge corpus
- Maintain sovereignty over your data and compute resources

### 🔮 Neural Synthesis
Conversations contextualized with vector-embedded knowledge, transforming static data into neural pathways.

### 🖥️ Bring Your Own Silicon
BYOG, BYOLLM. Your hardware, your models, your sovereignty. Local-first, future-proof computing paradigm powered by [Ollama](https://ollama.ai/).

### 🧩 Context Enrichment
Model Control Protocol integrations enhance cognitive abilities, extending the neural fabric beyond traditional boundaries.

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- Milvus vector database (can be run via Docker)
- [Ollama](https://ollama.ai/) for local LLM inference

### Installation
1. Clone this repository
```bash
git clone https://github.com/yourusername/thoughtfull.git
cd thoughtfull
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install and start Milvus following the instructions at [milvus.io](https://milvus.io/)

4. Install and start Ollama following the instructions at [ollama.ai](https://ollama.ai/)

5. Copy the example environment file and configure it:
```bash
cp .env.example .env
```

6. Start the application:
```bash
uvicorn app.main:app --reload
```

## 🐳 Docker Deployment

ThoughtFull can be deployed using Docker for easy setup and management.

### Docker Image

The official ThoughtFull Docker image is available at `ghcr.io/kanshi/thoughtfull:1.0.0`.

### Directory Structure

When deploying with Docker, you'll need to create two local directories to persist data:

- **Logs Directory**: Stores JSON-formatted log files for monitoring and debugging
- **Data Directory**: Stores the application's data, including document storage

### Running with Docker

```bash
docker run -d \
  -p 8000:8000 \
  -v /opt/thoughtfull/logs:/app/logs \
  -v /opt/thoughtfull/data:/app/data \
  -e DEFAULT_LLM_MODEL=llama3.2 \
  -e MILVUS_HOST=milvus-standalone \
  -e OLLAMA_BASE_URL=http://127.0.0.1:11434 \
  --network=milvus \
  --name thoughtfull \
  ghcr.io/kanshi/thoughtfull:1.0.0
```

### Docker Network Configuration

When running Milvus with Docker Compose, it creates a Docker network named `milvus`. To enable ThoughtFull to communicate with Milvus, you must connect the ThoughtFull container to this network using the `--network=milvus` flag.

**Important Notes:**

- The `MILVUS_HOST` should be set to the service name of the Milvus container (typically `milvus-standalone` when using Docker Compose)
- For Ollama integration, set the `OLLAMA_BASE_URL` to point to the Ollama instance (e.g., `http://127.0.0.1:11434` if running Ollama on the host)
- Mount volumes for logs (`/app/logs`) and data (`/app/data`) to persist information between container restarts

## 💬 Usage

1. Visit `http://localhost:8000` to access the ThoughtFull interface
2. Upload your documents to create your neural knowledge base
3. Enter The Matrix to begin conversing with your contextualized knowledge

For API details, visit the Swagger UI documentation at `http://localhost:8000/docs`

## 🗄️ Milvus Configuration

ThoughtFull uses [Milvus](https://milvus.io/) as its vector database to store and search document embeddings. Configure your Milvus instance in the `.env` file:

```
# Milvus server connection
MILVUS_HOST=192.168.3.11  # Default, change to your Milvus server address
MILVUS_PORT=19530        # Default Milvus port

# Collection configuration
COLLECTION_NAME=document_store            # For document embeddings
CONVERSATION_COLLECTION=conversation_history  # For conversation history
```

Additional vector-related settings:

```
# Vector dimensions for embeddings
VECTOR_DIMENSION=384  # Based on the all-MiniLM-L6-v2 model

# Embedding model configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Sentence transformer model used for embeddings
```

ThoughtFull automatically creates the necessary collections on startup if they don't exist. The collections are structured to efficiently store and retrieve both document chunks and conversation history for contextual responses.

## 🔗 Ollama Integration

ThoughtFull leverages [Ollama](https://ollama.ai/) to provide:
- Local LLM inference without sending data to third-party APIs
- Support for a variety of open-source models
- Complete control over your AI stack
- Customizable model parameters for optimal performance

Configure your preferred Ollama model in the `.env` file:
```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

You might need to download the model from [Ollama](https://ollama.ai/) first. By default thoughtless uses llama3.2, so to get that model available in your local ollama instance run:
```
ollama pull llama3.2
```
If you pull other models, you'll be able to switch throught them in the UI/api, and you can also update the default `OLLAMA_MODEL` variable in the `.env` file.

## 🧪 Philosophy

Built for the digital sovereignty movement - your data, your compute, your future. Each query traverses semantic networks, pulling context from your private dataverse, creating a synesthetic symphony of meaning.

---

*The neural revolution starts with you.*
