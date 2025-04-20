"""
Context builder step for the chat context pipeline
"""
from typing import Dict, Any, List
import json
from app.pipeline.pipeline import PipelineStep

class ContextBuilderStep(PipelineStep):
    """
    Builds a comprehensive context from various inputs for LLM consumption
    """
    
    def __init__(self, max_context_items: int = 5, include_scores: bool = False, 
                 min_score: float = 0.7, max_conversation_items: int = 3):
        """
        Initialize the context builder step
        
        Args:
            max_context_items: Maximum number of context items to include
            include_scores: Whether to include relevance scores in context
            min_score: Minimum relevance score for context items
            max_conversation_items: Maximum number of conversation items to include
        """
        self.max_context_items = max_context_items
        self.include_scores = include_scores
        self.min_score = min_score
        self.max_conversation_items = max_conversation_items
    
    def format_document_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format document search results into a readable string
        
        Args:
            results: List of document search result dictionaries
            
        Returns:
            Formatted string of document results
        """
        if not results:
            return ""
            
        formatted_results = []
        
        for i, result in enumerate(results[:self.max_context_items]):
            score_str = f" [Relevance: {result['score']:.2f}]" if self.include_scores else ""
            
            result_str = f"[{i+1}] From {result['file_name']}:{result['chunk_id']}{score_str}\n{result['content']}"
            formatted_results.append(result_str)
        
        return "\n\n".join(formatted_results)
    
    def format_conversation_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format conversation search results into a readable string
        
        Args:
            results: List of conversation search result dictionaries
            
        Returns:
            Formatted string of conversation history results
        """
        if not results:
            return ""
            
        formatted_results = []
        
        for i, result in enumerate(results[:self.max_conversation_items]):
            score_str = f" [Relevance: {result['score']:.2f}]" if self.include_scores else ""
            timestamp = result.get('timestamp', '').split('T')[0] # Just show the date part
            
            result_str = f"[{i+1}] From previous conversation on {timestamp}{score_str}\n{result['role']}: {result['content']}"
            formatted_results.append(result_str)
        
        return "\n\n".join(formatted_results)
    
    def format_search_results(self, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Format search results into formatted strings by type
        
        Args:
            results: List of mixed search result dictionaries
            
        Returns:
            Dictionary with formatted strings by result type
        """
        if not results:
            return {"documents": "", "conversations": ""}
        
        # Separate results by type
        document_results = [r for r in results if r.get('type') == 'document']
        conversation_results = [r for r in results if r.get('type') == 'conversation']
        
        # Format each type separately
        formatted_docs = self.format_document_results(document_results)
        formatted_convs = self.format_conversation_results(conversation_results)
        
        return {
            "documents": formatted_docs,
            "conversations": formatted_convs
        }
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the context to build a comprehensive context for the LLM
        
        Args:
            context: The current pipeline context
            
        Returns:
            Updated context with built context for LLM
        """
        # Initialize built context with an empty dictionary
        built_context = {}
        
        # Add the formatted user message
        if 'formatted_message' in context:
            built_context['current_message'] = context['formatted_message']
        
        # Add chat history if available
        if 'history' in context and context['history']:
            built_context['history'] = context['history']
        
        # Add system message if available
        if 'formatted_system_message' in context:
            built_context['system_message'] = context['formatted_system_message']
        
        # Filter and format search results
        search_results = context.get('search_results', [])
        filtered_results = [r for r in search_results if r.get('score', 0) >= self.min_score]
        
        if filtered_results:
            formatted_results_by_type = self.format_search_results(filtered_results)
            
            # Add document results if available
            if formatted_results_by_type["documents"]:
                built_context['document_context'] = formatted_results_by_type["documents"]
                doc_count = len([r for r in filtered_results if r.get('type') == 'document'])
                built_context['document_context_count'] = doc_count
            
            # Add conversation results if available
            if formatted_results_by_type["conversations"]:
                built_context['conversation_context'] = formatted_results_by_type["conversations"]
                conv_count = len([r for r in filtered_results if r.get('type') == 'conversation'])
                built_context['conversation_context_count'] = conv_count
            
            # Total context count
            built_context['search_context_count'] = len(filtered_results)
        
        # Add any additional metadata from the context
        if 'metadata' in context:
            built_context['metadata'] = context['metadata']
        
        # Add session information if available
        if 'session_id' in context:
            built_context['session_id'] = context['session_id']
        
        # Store the built context in the pipeline context
        context['built_context'] = built_context
        
        return context
