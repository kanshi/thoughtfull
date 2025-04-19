import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any

from app.models.chat import ChatRequest, ChatResponse, ModelsResponse, ModelInfo
from app.services.ollama import OllamaService
from app.database import search_documents
from app.services.embedding import EmbeddingService
from app.config import (
    DEFAULT_LLM_MODEL, 
    CPU_LLM_MODEL, 
    GPU_LLM_MODEL, 
    MAX_CONTEXT_CHUNKS, 
    MAX_HISTORY_LENGTH
)

router = APIRouter(prefix="/chat", tags=["chat"])

# In-memory store of chat sessions - in a production app, use a database
chat_sessions = {}

def get_ollama_service(model: Optional[str] = None) -> OllamaService:
    """
    Dependency that provides an OllamaService instance
    
    Args:
        model: Optional model name to use
        
    Returns:
        OllamaService instance
    """
    return OllamaService(model=model or DEFAULT_LLM_MODEL)


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """
    Chat endpoint for interacting with the LLM
    
    - **message**: User message to send to the LLM
    - **session_id**: (Optional) Session ID to continue a conversation
    - **model**: (Optional) LLM model to use for this message
    - **search_query**: (Optional) Explicit search query if different from the message
    - **include_context**: Whether to include search results in context
    
    Returns a response from the LLM with conversation info
    """
    # Override service model if specified in request
    if request.model:
        ollama_service.model = request.model
    
    # Get or create a session
    session_id = request.session_id or str(uuid.uuid4())
    
    # Get or initialize context
    context = chat_sessions.get(session_id, [])
    
    # Search query is either explicitly provided or is the message itself
    search_query = request.search_query or request.message
    search_results = []
    
    if request.include_context:
        # Get embeddings for search
        embedding_service = EmbeddingService()
        query_embedding = embedding_service.get_embedding(search_query)
        
        # Search in vector database
        search_hits = search_documents(query_embedding, MAX_CONTEXT_CHUNKS)
        
        # Format search results
        for hits in search_hits:
            for hit in hits:
                search_results.append({
                    "file_id": hit.entity.get('file_id'),
                    "file_name": hit.entity.get('file_name'),
                    "content": hit.entity.get('content'),
                    "chunk_id": hit.entity.get('chunk_id'),
                    "score": float(hit.score)
                })
    
    # Get response from LLM
    response_data = ollama_service.search_and_respond(
        query=request.message,
        search_results=search_results,
        context=context[-MAX_HISTORY_LENGTH:] if context else None
    )
    
    # Extract the response
    if "error" in response_data:
        raise HTTPException(status_code=500, detail=response_data["error"])
    
    assistant_message = response_data.get("message", {}).get("content", "")
    
    # Update context with new messages
    context.append({"role": "user", "content": request.message})
    context.append({"role": "assistant", "content": assistant_message})
    
    # Store updated context
    chat_sessions[session_id] = context
    
    return ChatResponse(
        response=assistant_message,
        session_id=session_id,
        search_results=search_results if request.include_context else None,
        model=ollama_service.model
    )


@router.get("/models")
async def get_models(session_id: Optional[str] = None):
    """
    Get available LLM models from Ollama
    
    Returns a list of available models and currently selected models
    """
    models = OllamaService.get_available_models()
    
    model_infos = []
    for model in models:
        model_info = ModelInfo(
            name=model.get("name", ""),
            size=model.get("size"),
            modified_at=model.get("modified_at"),
            digest=model.get("digest"),
            is_current=(model.get("name") == DEFAULT_LLM_MODEL)
        )
        model_infos.append(model_info)
    
    return ModelsResponse(
        models=model_infos,
        current_model=DEFAULT_LLM_MODEL,
        cpu_model=CPU_LLM_MODEL,
        gpu_model=GPU_LLM_MODEL
    )


@router.get("/models/switch")
async def switch_model(model_name: str = Query(..., description="Model name to switch to"), session_id: Optional[str] = None):
    """
    Switch the active LLM model
    
    - **model_name**: Name of the model to switch to
    - **session_id**: Optional session ID to update the model for a specific session
    
    Returns confirmation of model switch
    """
    global DEFAULT_LLM_MODEL
    
    # Check if model exists
    models = OllamaService.get_available_models()
    model_names = [model.get("name") for model in models]
    
    if model_name not in model_names:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_name}' not found. Available models: {', '.join(model_names)}"
        )
    
    # Update the default model
    DEFAULT_LLM_MODEL = model_name
    
    # If we have a session, update its model too
    if session_id and session_id in chat_sessions:
        # Update the model in any assistant messages in the history
        pass
    
    return {"success": True, "message": f"Switched to model: {model_name}", "model": model_name}


@router.get("/sessions/{session_id}", response_model=List[Dict[str, Any]])
async def get_session(session_id: str):
    """
    Get chat history for a session
    
    - **session_id**: ID of the session to retrieve
    
    Returns the chat history for the specified session
    """
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    
    return chat_sessions[session_id]


@router.delete("/sessions/{session_id}", response_model=Dict[str, str])
async def delete_session(session_id: str):
    """
    Delete a chat session
    
    - **session_id**: ID of the session to delete
    
    Returns confirmation of deletion
    """
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    
    del chat_sessions[session_id]
    
    return {"status": "success", "message": f"Session '{session_id}' deleted"}
