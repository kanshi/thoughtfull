"""
Vector search step for the chat context pipeline
"""
from typing import Dict, Any, List, Optional
import logging
import numpy as np
from app.pipeline.pipeline import PipelineStep
from app.services.embedding import EmbeddingService
from app.database import search_documents, search_conversations
from app.config import MAX_CONTEXT_CHUNKS

class VectorSearchStep(PipelineStep):
    """
    Performs vector search in the Milvus database using message embeddings,
    including both document search and conversation history search
    """
    
    def __init__(self, max_results: int = MAX_CONTEXT_CHUNKS, 
                 score_threshold: float = 0.6,
                 include_conversations: bool = True,
                 conversation_results_ratio: float = 0.5):
        """
        Initialize the vector search step
        
        Args:
            max_results: Maximum number of search results to include
            score_threshold: Minimum similarity score threshold for results
            include_conversations: Whether to include past conversations in search
            conversation_results_ratio: Ratio of conversation results to document results
        """
        self.max_results = max_results
        self.score_threshold = score_threshold
        self.include_conversations = include_conversations
        self.conversation_results_ratio = conversation_results_ratio
        self.embedding_service = EmbeddingService()
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text string
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.embedding_service.get_embedding(text)
    
    def search_documents(self, query_embedding: np.ndarray, max_results: int) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity
        
        Args:
            query_embedding: Query embedding vector
            max_results: Maximum number of results to return
            
        Returns:
            List of document search results
        """
        # Search in documents vector database
        search_hits = search_documents(query_embedding, max_results)
        
        # Format search results
        results = []
        for hits in search_hits:
            for hit in hits:
                score = float(hit.score)
                
                # Only include results above the threshold
                if score >= self.score_threshold:
                    results.append({
                        "type": "document",
                        "file_id": hit.entity.get('file_id'),
                        "file_name": hit.entity.get('file_name'),
                        "content": hit.entity.get('content'),
                        "chunk_id": hit.entity.get('chunk_id'),
                        "score": score
                    })
        
        return results
    
    def search_conversation_history(self, query_embedding: np.ndarray, max_results: int) -> List[Dict[str, Any]]:
        """
        Search for similar conversation messages using vector similarity
        
        Args:
            query_embedding: Query embedding vector
            max_results: Maximum number of results to return
            
        Returns:
            List of conversation search results
        """
        logging.info(f"=== SEARCHING CONVERSATION HISTORY ===")
        logging.info(f"Requesting up to {max_results} conversation results with threshold {self.score_threshold}")
        
        # Search in conversation history vector database
        search_hits = search_conversations(query_embedding, max_results)
        
        # Log raw search hits info
        hit_count = sum(len(hits) for hits in search_hits)
        logging.info(f"Raw conversation search returned {hit_count} total hits")
        
        # Format search results
        results = []
        filtered_count = 0
        for i, hits in enumerate(search_hits):
            for hit in hits:
                score = float(hit.score)
                session_id = hit.entity.get('session_id')
                role = hit.entity.get('role')
                content_preview = hit.entity.get('content', '')[:50]
                
                # Log each hit regardless of threshold
                log_level = logging.INFO if score >= self.score_threshold else logging.DEBUG
                logging.log(log_level, f"Conversation hit: session={session_id}, role={role}, "
                            f"score={score:.4f}, content='{content_preview}...'")
                
                # Only include results above the threshold
                if score >= self.score_threshold:
                    results.append({
                        "type": "conversation",
                        "session_id": session_id,
                        "role": role,
                        "content": hit.entity.get('content'),
                        "timestamp": hit.entity.get('timestamp'),
                        "chunk_id": hit.entity.get('chunk_id'),
                        "score": score
                    })
                else:
                    filtered_count += 1
        
        logging.info(f"Conversation search: {len(results)} results kept, {filtered_count} filtered out by threshold")
        logging.info(f"=== CONVERSATION SEARCH COMPLETE ===\n")
        return results
        
    def search(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """
        Search for similar content using vector similarity
        across both documents and conversation history
        
        Args:
            query_embedding: Query embedding vector
            
        Returns:
            List of combined search results
        """
        results = []
        
        # Calculate how many results to get from each source
        doc_results_count = self.max_results
        conv_results_count = 0
        
        if self.include_conversations:
            conv_results_count = int(self.max_results * self.conversation_results_ratio)
            doc_results_count = self.max_results - conv_results_count
        
        logging.info(f"Vector search configured for: {doc_results_count} document results, {conv_results_count} conversation results")
        
        # Get document search results
        try:
            doc_results = self.search_documents(query_embedding, doc_results_count)
            results.extend(doc_results)
            logging.info(f"Document search returned {len(doc_results)} results")
        except Exception as e:
            logging.error(f"Error during document search: {str(e)}")
            doc_results = []
        
        # Get conversation search results if enabled
        if self.include_conversations and conv_results_count > 0:
            try:
                conv_results = self.search_conversation_history(query_embedding, conv_results_count)
                logging.info(f"Conversation search returned {len(conv_results)} results")
                results.extend(conv_results)
            except Exception as e:
                logging.error(f"Error during conversation search: {str(e)}")
                logging.info("Continuing without conversation results due to error")
        
        # Sort all results by relevance score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Log the breakdown of results
        doc_count = len([r for r in results if r.get('type') == 'document'])
        conv_count = len([r for r in results if r.get('type') == 'conversation'])
        logging.info(f"Final combined results: {len(results)} total ({doc_count} documents, {conv_count} conversations)")
        
        return results
    
    def select_search_query(self, context: Dict[str, Any]) -> str:
        """
        Select the best search query from the context
        
        Args:
            context: The current context
            
        Returns:
            Search query string
        """
        # Use explicit search_query if provided
        if 'search_query' in context and context['search_query']:
            return context['search_query']
        
        # Use the most relevant n-grams if available
        if 'ngrams' in context and context['ngrams']:
            # For simplicity, use the longest n-gram as it may be most specific
            return max(context['ngrams'], key=len)
        
        # Fall back to the original message
        return context['message']
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the context to perform vector search
        
        Args:
            context: The current pipeline context
            
        Returns:
            Updated context with search results
        """
        # Log start of vector search process
        session_id = context.get('session_id', 'unknown_session')
        logging.info(f"=== STARTING VECTOR SEARCH PROCESSING FOR SESSION {session_id} ===")
        
        # Get the search query
        search_query = self.select_search_query(context)
        logging.info(f"Selected search query: '{search_query[:100]}...'")
        
        # Generate embedding for search
        logging.info("Generating embedding for search query")
        query_embedding = self.get_embedding(search_query)
        logging.info(f"Generated embedding with shape {query_embedding.shape}")
        
        # Store the embedding for potential future use
        context['message_embedding'] = query_embedding
        
        # Log search configuration
        logging.info(f"Search configuration: include_conversations={self.include_conversations}, "
                     f"conversation_results_ratio={self.conversation_results_ratio}, "
                     f"score_threshold={self.score_threshold}")
        
        # Perform search
        search_results = self.search(query_embedding)
        
        # Log search results summary
        doc_count = len([r for r in search_results if r.get('type') == 'document'])
        conv_count = len([r for r in search_results if r.get('type') == 'conversation'])
        logging.info(f"Search returned {len(search_results)} total results: {doc_count} documents, {conv_count} conversations")
        
        # Add search results to context
        context['search_results'] = search_results
        
        # Log completion of vector search process
        logging.info(f"=== VECTOR SEARCH PROCESSING COMPLETE FOR SESSION {session_id} ===\n")
        return context
