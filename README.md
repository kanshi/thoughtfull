# ThoughtFull - Semantic Search for Private Documents

A semantic search service built with FastAPI and Milvus vector database that allows you to:
- Upload private documents (PDF, TXT)
- Process and embed documents into vector space
- Perform semantic searches across your document collection

## Setup

### Prerequisites
- Python 3.8+
- Milvus database (can be run via Docker)

### Installation
1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Start Milvus (using Docker or whatever you like):

4. Copy the example environment file and configure it:
```bash
cp .env.example .env
```
5. Start the application:
```bash
uvicorn app.main:app --reload
```

## API Usage

Visit the Swagger UI documentation at http://localhost:8000/docs for complete API details.
