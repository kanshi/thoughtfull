import logging
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, VECTOR_DIMENSION, CONVERSATION_COLLECTION

def connect_to_milvus():
    # Print the configuration values before connecting
    logging.info(f"Attempting to connect to Milvus using host: {MILVUS_HOST}, port: {MILVUS_PORT}")
    
    connections.connect(
        alias="default", 
        host=MILVUS_HOST, 
        port=MILVUS_PORT
    )
    logging.info(f"Connected to Milvus server at {MILVUS_HOST}:{MILVUS_PORT}")

def create_collection_if_not_exists():
    if utility.has_collection(COLLECTION_NAME):
        logging.info(f"Collection '{COLLECTION_NAME}' already exists")
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
    logging.info(f"Created collection '{COLLECTION_NAME}' and index")
    
    return collection


def create_conversation_collection_if_not_exists():
    """Create a collection for storing conversation history if it doesn't exist"""
    if utility.has_collection(CONVERSATION_COLLECTION):
        logging.info(f"Collection '{CONVERSATION_COLLECTION}' already exists")
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
    logging.info(f"Created collection '{CONVERSATION_COLLECTION}' and index")
    
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
        logging.info(f"Dropping existing collection '{COLLECTION_NAME}'")
        utility.drop_collection(COLLECTION_NAME)
        logging.info(f"Collection '{COLLECTION_NAME}' dropped successfully")
    
    # Create new collection with current configuration
    return create_collection_if_not_exists()


def recreate_conversation_collection():
    """Drop the existing conversation collection and recreate it"""
    connect_to_milvus()
    
    # Check if collection exists and drop it
    if utility.has_collection(CONVERSATION_COLLECTION):
        logging.info(f"Dropping existing collection '{CONVERSATION_COLLECTION}'")
        utility.drop_collection(CONVERSATION_COLLECTION)
        logging.info(f"Collection '{CONVERSATION_COLLECTION}' dropped successfully")
    
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
        logging.info(f"Inserting {num_chunks} chunks with file_id: {file_id}, file_name: {file_name}")
        mr = collection.insert(data)
        logging.info(f"Insert result: {mr}")
    except Exception as e:
        logging.error(f"Error during insertion: {str(e)}")
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


def store_conversation_message(session_id: str, role: str, content: str, embedding: np.ndarray, sequence: int, metadata: Optional[Dict[str, Any]] = None, related_message: Optional[Dict[str, Any]] = None):
    """
    Store a conversation message in Milvus
    
    Args:
        session_id: The session ID for the conversation
        role: The role of the message sender (user or assistant)
        content: The content of the message
        embedding: The vector embedding of the message
        sequence: The sequence number of the message in the conversation
        metadata: Optional metadata for the message
        related_message: Optional related message (the message this is responding to or the response to this message)
    """
    logging.info(f"=== STORING CONVERSATION MESSAGE ===")
    logging.info(f"Session ID: {session_id}")
    logging.info(f"Role: {role}")
    logging.info(f"Sequence: {sequence}")
    logging.info(f"Content preview: '{content[:50]}...'")
    if related_message:
        logging.info(f"Related message role: {related_message.get('role')}")
        logging.info(f"Related message preview: '{related_message.get('content', '')[:50]}...'")
    
    # Get collection and prepare data for insertion
    try:
        # Get collection
        collection = get_conversation_collection()
        
        # Convert embedding to numpy array if not already
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
            
        logging.info(f"Embedding shape: {embedding.shape}")
        
        # Generate timestamp
        timestamp = datetime.now().isoformat()
        
        # Generate a unique chunk ID
        chunk_id = f"{session_id}_{sequence}"
        logging.info(f"Generated chunk_id: {chunk_id}")
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
            
        # Add related message to metadata if provided
        if related_message:
            metadata["related_message"] = related_message
        
        # Prepare data for insertion
        # Fields must match the order defined in create_conversation_collection_if_not_exists
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
        
        # Log field counts for debugging
        logging.info(f"Data field count: {len(data)}")
        logging.info(f"Executing Milvus insert for conversation message")
        mr = collection.insert(data)
        logging.info(f"Insert result: {mr}")
        
        # Flush to make sure data is persisted
        logging.info("Flushing collection to ensure data persistence")
        collection.flush()
        
        # Verify the insertion by trying to retrieve the entry
        try:
            # Make sure collection is loaded before query
            try:
                collection.load()
                logging.info("Collection loaded for verification query")
            except Exception as load_e:
                logging.error(f"Warning: Could not load collection for verification: {str(load_e)}")
                # Continue anyway as this is just verification
            
            expr = f'chunk_id == "{chunk_id}"'
            verify_results = collection.query(expr=expr, output_fields=["session_id", "role"])
            if verify_results:
                logging.info(f"Successfully verified message insertion: {verify_results}")
            else:
                logging.warning(f"Could not verify message insertion for chunk_id={chunk_id}")
        except Exception as verify_e:
            logging.error(f"Error verifying message insertion: {str(verify_e)}")
            logging.info("Continuing despite verification error - message was likely stored successfully")
        
        logging.info(f"=== CONVERSATION MESSAGE STORED SUCCESSFULLY ===\n")
        return True
    except Exception as e:
        logging.error(f"Error storing conversation message: {str(e)}", exc_info=True)
        return False


def search_conversations(query_embedding, limit=5):
    """Search for similar conversation messages based on vector similarity"""
    logging.info(f"Starting conversation search with limit={limit}")
    
    # Return empty results if there's an error
    empty_results = []
    
    try:
        # Connect to Milvus
        try:
            connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
            logging.info(f"Connected to Milvus for conversation search")
        except Exception as conn_e:
            logging.error(f"Failed to connect to Milvus: {str(conn_e)}")
            return empty_results
        
        # Check if collection exists
        if not utility.has_collection(CONVERSATION_COLLECTION):
            logging.warning(f"Collection '{CONVERSATION_COLLECTION}' does not exist")
            return empty_results
            
        # Get collection
        try:
            collection = Collection(CONVERSATION_COLLECTION)
            logging.info(f"Successfully got conversation collection")
        except Exception as coll_e:
            logging.error(f"Failed to get conversation collection: {str(coll_e)}")
            return empty_results
        
        # Load collection
        try:
            collection.load()
            logging.info("Successfully loaded conversation collection")
        except Exception as load_e:
            logging.error(f"Error loading conversation collection: {str(load_e)}")
            return empty_results
        
        # Configure search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 32}
        }
        
        # Execute search with error handling
        try:
            logging.info(f"Executing Milvus search for conversations with limit={limit}")
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["chunk_id", "session_id", "timestamp", "role", "content", "sequence", "metadata"]
            )
            
            # Log results summary
            total_hits = sum(len(hits) for hits in results)
            logging.info(f"Conversation search returned {total_hits} total hits across {len(results)} results")
            
            # Skip detailed logging to avoid Entity.get() errors
            logging.debug(f"Found {sum(len(hits) for hits in results)} total hits across {len(results)} results")
            
            return results
        except Exception as search_e:
            logging.error(f"Error executing conversation search: {str(search_e)}")
            return empty_results
            
    except Exception as e:
        logging.error(f"Unexpected error in search_conversations: {str(e)}", exc_info=True)
        return empty_results


def get_conversation_history(session_id: str):
    """Get all messages for a specific conversation session"""
    logging.info(f"Getting conversation history for session {session_id}")
    collection = get_conversation_collection()
    
    # Make sure collection is loaded before query
    try:
        collection.load()
        logging.info("Successfully loaded collection for conversation history retrieval")
    except Exception as e:
        logging.error(f"Error loading collection for conversation history: {str(e)}")
        # Try to continue anyway
    
    try:
        # Query by session_id
        expr = f'session_id == "{session_id}"'
        results = collection.query(
            expr=expr,
            output_fields=["chunk_id", "session_id", "timestamp", "role", "content", "sequence", "metadata"],
        )
        
        logging.info(f"Retrieved {len(results)} messages for session {session_id}")
        
        # Sort by sequence number
        results.sort(key=lambda x: x['sequence'])
        
        # Log result details for debugging
        for i, msg in enumerate(results):
            logging.info(f"History message {i+1}: role={msg.get('role')}, sequence={msg.get('sequence')}, "
                        f"content_preview='{msg.get('content', '')[:50]}...'")
        
        return results
    except Exception as e:
        logging.error(f"Error retrieving conversation history: {str(e)}", exc_info=True)
        return []
