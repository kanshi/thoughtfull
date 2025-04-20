from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class ChatMessage(BaseModel):
    """Chat message model for both user inputs and AI responses"""
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the message")


class ChatSession(BaseModel):
    """Chat session model that contains messages and metadata"""
    id: str = Field(..., description="Unique identifier for the chat session")
    messages: List[ChatMessage] = Field(default_factory=list, description="List of chat messages")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    model: str = Field(..., description="LLM model being used for this session")


class ChatRequest(BaseModel):
    """Chat request model used for API endpoints"""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for continuing a conversation")
    model: Optional[str] = Field(None, description="LLM model to use (overrides default)")
    search_query: Optional[str] = Field(None, description="Optional explicit search query")
    include_context: bool = Field(True, description="Whether to include search results in context")


class ChatResponse(BaseModel):
    """Chat response model returned by the API"""
    response: str = Field(..., description="AI response message")
    session_id: str = Field(..., description="Session ID for the conversation")
    search_results: Optional[List[Dict[str, Any]]] = Field(None, description="Relevant search results")
    document_context: Optional[List[Dict[str, Any]]] = Field(None, description="Document context used for prompting")
    conversation_context: Optional[List[Dict[str, Any]]] = Field(None, description="Conversation context used for prompting")
    model: str = Field(..., description="LLM model used for response")
    
    
class ModelInfo(BaseModel):
    """Model information returned by the models endpoint"""
    name: str = Field(..., description="Model name/identifier")
    size: Optional[int] = Field(None, description="Model size in bytes")
    modified_at: Optional[datetime] = Field(None, description="Last modified timestamp")
    digest: Optional[str] = Field(None, description="Model digest")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional model details")
    is_current: bool = Field(False, description="Whether this is the currently active model")


class ModelsResponse(BaseModel):
    """Response model for the available models endpoint"""
    models: List[ModelInfo] = Field(..., description="List of available models")
    current_model: str = Field(..., description="Currently selected model name")
    cpu_model: str = Field(..., description="Configured CPU-optimized model")
    gpu_model: str = Field(..., description="Configured GPU-optimized model")
