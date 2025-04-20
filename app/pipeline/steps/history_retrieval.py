"""
Chat history retrieval step for the chat context pipeline
"""
from typing import Dict, Any, List, Optional
from app.pipeline.pipeline import PipelineStep
from app.config import MAX_HISTORY_LENGTH

class ChatHistoryStep(PipelineStep):
    """
    Retrieves and processes chat history for context enrichment
    """
    
    def __init__(self, max_history: int = MAX_HISTORY_LENGTH, include_timestamps: bool = True):
        """
        Initialize the chat history step
        
        Args:
            max_history: Maximum number of history messages to include
            include_timestamps: Whether to include timestamps in context
        """
        self.max_history = max_history
        self.include_timestamps = include_timestamps
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the context to retrieve chat history
        
        Args:
            context: The current pipeline context
            
        Returns:
            Updated context with chat history
        """
        # Skip if no session_id is provided
        if 'session_id' not in context:
            context['history'] = []
            return context
        
        session_id = context['session_id']
        
        # Get history from in-memory store or database
        # For now, assume it's passed in the context or use empty list
        history = context.get('raw_history', [])
        
        # Limit history to max_history
        limited_history = history[-self.max_history:] if history else []
        
        # Process and format history if needed
        processed_history = []
        for message in limited_history:
            # Ensure each history item has required fields
            if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                continue
                
            # Copy message to avoid modifying original
            processed_message = message.copy()
            
            # Remove timestamps if not needed
            if not self.include_timestamps and 'timestamp' in processed_message:
                del processed_message['timestamp']
                
            processed_history.append(processed_message)
        
        # Add processed history to context
        context['history'] = processed_history
        
        return context
