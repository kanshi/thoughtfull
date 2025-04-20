"""
Context Manager Service for managing session contexts across the application
"""
from typing import Dict, Any

# Singleton for global context management
_session_contexts: Dict[str, Dict[str, Any]] = {}

def get_session_context(session_id: str) -> Dict[str, Any]:
    """
    Get the context data for a specific session
    
    Args:
        session_id: The session ID to get context for
        
    Returns:
        The context data for the session or empty dict if not found
    """
    return _session_contexts.get(session_id, {
        "document_context": [],
        "conversation_context": []
    })

def set_session_context(session_id: str, context_data: Dict[str, Any]) -> None:
    """
    Set the context data for a specific session
    
    Args:
        session_id: The session ID to set context for
        context_data: The context data to store
    """
    _session_contexts[session_id] = context_data

def get_all_session_contexts() -> Dict[str, Dict[str, Any]]:
    """
    Get all session contexts
    
    Returns:
        Dictionary of all session contexts
    """
    return _session_contexts
