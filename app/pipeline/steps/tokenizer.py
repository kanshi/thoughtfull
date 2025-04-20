"""
Tokenization step for the chat context pipeline
"""
from typing import Dict, Any, List
import re
from app.pipeline.pipeline import PipelineStep

class TokenizationStep(PipelineStep):
    """
    Tokenizes the user message into words and phrases for better search results
    """
    
    def __init__(self, min_token_length: int = 3, 
                 exclude_stopwords: bool = True, 
                 include_ngrams: bool = True,
                 max_ngram_size: int = 3):
        """
        Initialize the tokenization step
        
        Args:
            min_token_length: Minimum length of tokens to keep
            exclude_stopwords: Whether to exclude common stopwords
            include_ngrams: Whether to generate n-grams from tokens
            max_ngram_size: Maximum size of n-grams to generate
        """
        self.min_token_length = min_token_length
        self.exclude_stopwords = exclude_stopwords
        self.include_ngrams = include_ngrams
        self.max_ngram_size = max_ngram_size
        
        # Common English stopwords
        self.stopwords = set([
            "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
            "which", "this", "that", "these", "those", "then", "just", "so", "than",
            "such", "when", "who", "how", "where", "why", "is", "am", "are", "was",
            "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "can", "could", "will", "would", "shall", "should", "may", "might", "must",
            "to", "of", "in", "for", "on", "by", "at", "with", "about", "against",
            "between", "into", "through", "during", "before", "after", "above", "below",
            "from", "up", "down", "out", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "all", "any", "both", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "too", "very", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", 
            "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
            "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
            "theirs", "themselves"
        ])
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Filter by length and stopwords
        filtered_tokens = []
        for token in tokens:
            if len(token) >= self.min_token_length:
                if not self.exclude_stopwords or token not in self.stopwords:
                    filtered_tokens.append(token)
        
        return filtered_tokens
    
    def generate_ngrams(self, tokens: List[str]) -> List[str]:
        """
        Generate n-grams from tokens
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of n-grams
        """
        ngrams = []
        
        # Add individual tokens
        ngrams.extend(tokens)
        
        # Generate n-grams of sizes 2 to max_ngram_size
        for n in range(2, min(self.max_ngram_size + 1, len(tokens) + 1)):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i+n])
                ngrams.append(ngram)
                
        return ngrams
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the user message to extract tokens and n-grams
        
        Args:
            context: Context dictionary containing the user message
            
        Returns:
            Updated context with tokens and n-grams
        """
        # Ensure the user message is in the context
        if 'message' not in context:
            raise ValueError("Context must contain a 'message' key")
        
        message = context['message']
        
        # Tokenize the message
        tokens = self.tokenize(message)
        
        # Generate n-grams if enabled
        if self.include_ngrams and tokens:
            ngrams = self.generate_ngrams(tokens)
        else:
            ngrams = tokens
        
        # Update context with tokens and n-grams
        context['tokens'] = tokens
        context['ngrams'] = ngrams
        
        return context
