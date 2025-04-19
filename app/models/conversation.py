from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


class ConversationChunk(BaseModel):
    """Model for storing individual conversation chunks in Milvus"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the chunk")
    session_id: str = Field(..., description="Session ID the chunk belongs to")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the chunk")
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
    sequence: int = Field(..., description="Sequence number within the conversation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    vector_id: Optional[str] = Field(None, description="Vector ID in Milvus")


class ConversationMetadata(BaseModel):
    """Model for storing conversation metadata"""
    session_id: str = Field(..., description="Unique identifier for the conversation")
    title: Optional[str] = Field(None, description="Title of the conversation")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    message_count: int = Field(default=0, description="Number of messages in the conversation")
    model: str = Field(..., description="LLM model used for the conversation")
    summary: Optional[str] = Field(None, description="AI-generated summary of the conversation")
