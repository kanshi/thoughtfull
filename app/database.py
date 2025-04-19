from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, VECTOR_DIMENSION, CONVERSATION_COLLECTION

def connect_to_milvus():
    # Print the configuration values before connecting
    print(f"Attempting to connect to Milvus using host: {MILVUS_HOST}, port: {MILVUS_PORT}")
    
    # Ensure we're using the correct values
    print(f"Config values directly from config module: {MILVUS_HOST}:{MILVUS_PORT}")
    
    connections.connect(
        alias="default", 
        host=MILVUS_HOST, 
        port=MILVUS_PORT
    )
    print(f"Connected to Milvus server at {MILVUS_HOST}:{MILVUS_PORT}")

def create_collection_if_not_exists():
    if utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists")
        return Collection(COLLECTION_NAME)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
    ]
    
    schema = CollectionSchema(fields=fields, description="Document chunks with embeddings")
    
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Created collection '{COLLECTION_NAME}' and index")
    
    return collection


def create_conversation_collection_if_not_exists():
    """Create a collection for storing conversation history if it doesn't exist"""
    if utility.has_collection(CONVERSATION_COLLECTION):
        print(f"Collection '{CONVERSATION_COLLECTION}' already exists")
        return Collection(CONVERSATION_COLLECTION)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="sequence", dtype=DataType.INT64),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
    ]
    
    schema = CollectionSchema(fields=fields, description="Conversation history with embeddings")
    
    collection = Collection(name=CONVERSATION_COLLECTION, schema=schema)
    
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Created collection '{CONVERSATION_COLLECTION}' and index")
    
    return collection

def get_collection():
    connect_to_milvus()
    return create_collection_if_not_exists()


def get_conversation_collection():
    """Get the conversation collection"""
    connect_to_milvus()
    return create_conversation_collection_if_not_exists()

def recreate_collection():
    """Drop the existing collection and recreate it with the current configuration"""
    connect_to_milvus()
    
    # Check if collection exists and drop it
    if utility.has_collection(COLLECTION_NAME):
        print(f"Dropping existing collection '{COLLECTION_NAME}'")
        utility.drop_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' dropped successfully")
    
    # Create new collection with current configuration
    return create_collection_if_not_exists()


def recreate_conversation_collection():
    """Drop the existing conversation collection and recreate it"""
    connect_to_milvus()
    
    # Check if collection exists and drop it
    if utility.has_collection(CONVERSATION_COLLECTION):
        print(f"Dropping existing collection '{CONVERSATION_COLLECTION}'")
        utility.drop_collection(CONVERSATION_COLLECTION)
        print(f"Collection '{CONVERSATION_COLLECTION}' dropped successfully")
    
    # Create new collection with current configuration
    return create_conversation_collection_if_not_exists()

def insert_documents(chunks, embeddings, file_id, file_name):
    collection = get_collection()
    
    # Convert embeddings to numpy array if not already
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings, dtype=np.float32)
    
    # Prepare data for insertion with proper types
    num_chunks = len(chunks)
    
    # Following the Milvus example format for data insertion
    # The order must match the order of fields in the collection schema
    data = [
        # id field is auto-generated, so we don't include it
        [str(file_id) for _ in range(num_chunks)],          # file_id (VARCHAR)
        [str(file_name) for _ in range(num_chunks)],        # file_name (VARCHAR)
        [i for i in range(num_chunks)],                     # chunk_id (INT64)
        [str(chunk) for chunk in chunks],                    # content (VARCHAR)
        embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings  # embedding (FLOAT_VECTOR)
    ]
    
    try:
        print(f"Inserting {num_chunks} chunks with file_id: {file_id}, file_name: {file_name}")
        mr = collection.insert(data)
        print(f"Insert result: {mr}")
    except Exception as e:
        print(f"Error during insertion: {str(e)}")
        raise
    
    # Ensure data is searchable immediately
    collection.flush()
    return len(chunks)

def search_documents(query_embedding, limit=5):
    """Search for similar documents based on vector similarity"""
    collection = get_collection()
    collection.load()
    
    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": 32}
    }
    
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=limit,
        output_fields=["file_id", "file_name", "content", "chunk_id"]
    )
    
    return results


def store_conversation_message(session_id: str, role: str, content: str, embedding: np.ndarray, sequence: int, metadata: Optional[Dict[str, Any]] = None):
    """Store a conversation message in Milvus"""
    collection = get_conversation_collection()
    
    # Convert embedding to numpy array if not already
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding, dtype=np.float32)
    
    # Generate timestamp
    timestamp = datetime.now().isoformat()
    
    # Generate a unique chunk ID
    chunk_id = f"{session_id}_{sequence}"
    
    # Prepare metadata
    if metadata is None:
        metadata = {}
    
    # Prepare data for insertion
    data = [
        # id field is auto-generated
        [chunk_id],                              # chunk_id
        [session_id],                           # session_id
        [timestamp],                            # timestamp
        [role],                                 # role
        [content],                              # content
        [sequence],                             # sequence
        [metadata],                             # metadata
        embedding.reshape(1, -1).tolist()       # embedding
    ]
    
    try:
        print(f"Storing conversation message for session {session_id}, sequence {sequence}")
        mr = collection.insert(data)
        print(f"Insert result: {mr}")
        collection.flush()
        return True
    except Exception as e:
        print(f"Error storing conversation message: {str(e)}")
        return False


def search_conversations(query_embedding, limit=5):
    """Search for similar conversation messages based on vector similarity"""
    collection = get_conversation_collection()
    collection.load()
    
    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": 32}
    }
    
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=limit,
        output_fields=["chunk_id", "session_id", "timestamp", "role", "content", "sequence", "metadata"]
    )
    
    return results


def get_conversation_history(session_id: str):
    """Get all messages for a specific conversation session"""
    collection = get_conversation_collection()
    
    # Query by session_id
    expr = f'session_id == "{session_id}"'
    results = collection.query(
        expr=expr,
        output_fields=["chunk_id", "session_id", "timestamp", "role", "content", "sequence", "metadata"],
    )
    
    # Sort by sequence number
    results.sort(key=lambda x: x['sequence'])
    
    return results
