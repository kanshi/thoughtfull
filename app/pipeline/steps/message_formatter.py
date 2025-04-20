"""
Message formatter step for the chat context pipeline
"""
from typing import Dict, Any, List
from app.pipeline.pipeline import PipelineStep

class MessageFormatterStep(PipelineStep):
    """
    Formats messages in a standard structure for the LLM
    """
    
    def __init__(self, include_metadata: bool = True):
        """
        Initialize the message formatter step
        
        Args:
            include_metadata: Whether to include metadata in formatted messages
        """
        self.include_metadata = include_metadata
    
    def format_user_message(self, message: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Format a user message
        
        Args:
            message: User message text
            metadata: Optional metadata
            
        Returns:
            Formatted message dictionary
        """
        formatted = {
            "role": "user",
            "content": message
        }
        
        if self.include_metadata and metadata:
            formatted["metadata"] = metadata
        
        return formatted
    
    def format_assistant_message(self, message: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Format an assistant message
        
        Args:
            message: Assistant message text
            metadata: Optional metadata
            
        Returns:
            Formatted message dictionary
        """
        formatted = {
            "role": "assistant",
            "content": message
        }
        
        if self.include_metadata and metadata:
            formatted["metadata"] = metadata
        
        return formatted
    
    def format_system_message(self, message: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Format a system message
        
        Args:
            message: System message text
            metadata: Optional metadata
            
        Returns:
            Formatted message dictionary
        """
        formatted = {
            "role": "system",
            "content": message
        }
        
        if self.include_metadata and metadata:
            formatted["metadata"] = metadata
        
        return formatted
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the context to format messages
        
        Args:
            context: The current pipeline context
            
        Returns:
            Updated context with formatted messages
        """
        # Format the current user message
        if 'message' in context:
            message_metadata = {
                "timestamp": context.get('timestamp'),
                "tokens": context.get('tokens', []),
                "search_results_count": len(context.get('search_results', []))
            }
            
            formatted_message = self.format_user_message(
                context['message'],
                metadata=message_metadata if self.include_metadata else None
            )
            
            context['formatted_message'] = formatted_message
        
        # Format system message if not already in context
        if 'system_message' in context and 'formatted_system_message' not in context:
            formatted_system = self.format_system_message(context['system_message'])
            context['formatted_system_message'] = formatted_system
        
        return context
