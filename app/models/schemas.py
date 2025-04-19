from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class DocumentBase(BaseModel):
    file_id: str
    file_name: str
    
class DocumentResponse(DocumentBase):
    id: str
    
class SearchQuery(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=100)
    
class SearchResult(BaseModel):
    file_id: str
    file_name: str
    content: str
    score: float
    chunk_id: int
    
class SearchResponse(BaseModel):
    results: List[SearchResult]
    count: int
    query: str
    
class UploadResponse(BaseModel):
    file_id: str
    file_name: str
    chunks: int
    message: str
    
class ErrorResponse(BaseModel):
    error: str
