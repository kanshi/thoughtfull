from sentence_transformers import SentenceTransformer
import numpy as np
from app.config import EMBEDDING_MODEL
from app.utils.logging import get_logger

class EmbeddingService:
    def __init__(self):
        """Initialize the embedding model"""
        self.logger = get_logger(__name__)
        self.logger.info(f"Initializing embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.logger.info(f"Embedding model initialized with dimension: {self.model.get_sentence_embedding_dimension()}")
        
    def get_embeddings(self, texts):
        """
        Generate embeddings for a list of text chunks
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        if not texts:
            self.logger.warning("Received empty list for embedding generation")
            return np.array([])
        
        self.logger.info(f"Generating embeddings for {len(texts)} text chunks")
        
        try:
            # Generate embeddings for all texts
            embeddings = self.model.encode(texts, show_progress_bar=False)
            self.logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            raise
    
    def get_embedding(self, text):
        """
        Generate embedding for a single text
        
        Args:
            text: String to embed
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        if not text:
            self.logger.warning("Received empty text for embedding generation")
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        self.logger.info(f"Generating embedding for text of length {len(text)} characters")
        
        try:
            embedding = self.model.encode(text, show_progress_bar=False)
            self.logger.info(f"Successfully generated embedding with shape {embedding.shape}")
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
            raise
