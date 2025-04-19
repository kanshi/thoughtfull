from fastapi import APIRouter, Request, Depends, HTTPException, Form, Query, Body
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pathlib
import os
import uuid
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Iterator

from app.models.schemas import SearchResponse, SearchResult
from app.models.chat import ChatRequest
from app.services.embedding import EmbeddingService
from app.services.ollama import OllamaService
from app.database import search_documents, store_conversation_message
from app.config import UPLOAD_DIR, CPU_LLM_MODEL, GPU_LLM_MODEL, DEFAULT_LLM_MODEL
from app.utils.logging import get_logger

router = APIRouter(tags=["web"])

# Setup templates
BASE_DIR = pathlib.Path(__file__).parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Setup logging
logger = get_logger(__name__)

# In-memory storage for document metadata (in a real app, this would be in a database)
documents_store: List[Dict[str, Any]] = []

# In-memory storage for chat sessions
web_chat_sessions: Dict[str, List[Dict[str, Any]]] = {}

# In-memory storage for active messages in each session
active_messages: Dict[str, str] = {}

# In-memory storage for session-specific model selections
session_models: Dict[str, str] = {}

@router.get("/documents/list", response_class=HTMLResponse)
async def list_documents(request: Request):
    """
    Get a list of uploaded documents for the web interface
    """
    return templates.TemplateResponse(
        "document_list.html",
        {"request": request, "documents": documents_store}
    )

@router.post("/documents/upload", response_class=HTMLResponse)
async def upload_document_web(request: Request):
    """
    Handle document upload from the web interface and return HTML response
    """
    from app.api.documents import upload_document
    
    try:
        # Get the form data
        form = await request.form()
        file = form.get("file")
        
        # Call the API endpoint
        result = await upload_document(file)
        
        # Store document metadata
        documents_store.append({
            "file_id": result.file_id,
            "file_name": result.file_name,
            "chunks": result.chunks
        })
        
        # Return HTML response
        return templates.TemplateResponse(
            "upload_result.html",
            {
                "request": request,
                "file_id": result.file_id,
                "message": result.message,
                "chunks": result.chunks
            }
        )
    except HTTPException as e:
        return templates.TemplateResponse(
            "upload_result.html",
            {"request": request, "error": e.detail}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "upload_result.html",
            {"request": request, "error": str(e)}
        )

@router.get("/search", response_class=HTMLResponse)
async def search_web(
    request: Request,
    query: str,
    limit: int = 5
):
    """
    Handle search from the web interface and return HTML results
    """
    try:
        # Generate embedding for query
        embedding_service = EmbeddingService()
        query_embedding = embedding_service.get_embedding(query)
        
        # Search in vector database
        results = search_documents(query_embedding, limit)
        
        formatted_results = []
        if results:
            for hits in results:
                for hit in hits:
                    formatted_results.append(SearchResult(
                        file_id=hit.entity.get('file_id'),
                        file_name=hit.entity.get('file_name'),
                        content=hit.entity.get('content'),
                        chunk_id=hit.entity.get('chunk_id'),
                        score=float(hit.score)
                    ))
        
        # Return HTML response
        return templates.TemplateResponse(
            "search_results.html",
            {
                "request": request,
                "results": formatted_results,
                "count": len(formatted_results),
                "query": query
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "search_results.html",
            {"request": request, "error": str(e), "query": query, "results": [], "count": 0}
        )


@router.get("/chat")
async def chat_page(request: Request):
    """
    Chat page for interacting with documents
    """
    # Generate a unique session ID for this chat session
    session_id = str(uuid.uuid4())
    
    # Initialize the session with an empty list
    web_chat_sessions[session_id] = []
    # Set the default model for this session
    session_models[session_id] = DEFAULT_LLM_MODEL
    logger.info(f"Created new chat session with ID: {session_id} using model: {DEFAULT_LLM_MODEL}")
    
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "session_id": session_id,
            "current_model": DEFAULT_LLM_MODEL,
            "cpu_model": CPU_LLM_MODEL,
            "gpu_model": GPU_LLM_MODEL
        }
    )


@router.post("/web/chat/message")
async def process_chat_message(
    request: Request,
    data: Dict[str, str] = Body(...)
):
    """
    Process a chat message and store it in the session without rendering HTML
    """
    try:
        message = data.get("message")
        session_id = data.get("session_id")
        
        if not message or not session_id:
            logger.error(f"Missing required parameters: message={message}, session_id={session_id}")
            return JSONResponse(
                status_code=400,
                content={"error": "Missing required parameters: message and session_id"}
            )
        
        logger.info(f"Processing chat message for session {session_id}: '{message[:50]}...'")
        
        # Get context from session or initialize
        if session_id not in web_chat_sessions:
            logger.warning(f"Session {session_id} not found, creating new session")
            web_chat_sessions[session_id] = []
            
        context = web_chat_sessions[session_id]
        
        # Update context with user message
        user_message = {"role": "user", "content": message}
        context.append(user_message)
        
        # Store updated context
        web_chat_sessions[session_id] = context
        logger.info(f"Updated context for session {session_id}, new length: {len(context)}")
        
        # Store the current message as the active message for this session
        active_messages[session_id] = message
        logger.info(f"Set active message for session {session_id}: '{message[:50]}...'")
        
        # Start a task to store the user message in Milvus
        async def store_user_message_async():
            try:
                embedding_service = EmbeddingService()
                user_embedding = embedding_service.get_embedding(message)
                
                # Store user message with sequence number based on context length
                sequence = len(context) - 1  # 0-indexed
                store_conversation_message(
                    session_id=session_id,
                    role="user",
                    content=message,
                    embedding=user_embedding,
                    sequence=sequence,
                    metadata={"model": DEFAULT_LLM_MODEL}
                )
            except Exception as e:
                logger.error(f"Error storing user message in background: {str(e)}", exc_info=e)
        
        # Start the storage task without awaiting it
        asyncio.create_task(store_user_message_async())
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Message processed successfully",
                "session_id": session_id
            }
        )
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}", exc_info=e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process message: {str(e)}"}
        )

@router.post("/web/chat", response_class=HTMLResponse)
async def chat_message(
    request: Request,
    message: str = Form(...),
    session_id: str = Form(...)
):
    """
    Handle chat messages from the web interface
    """
    try:
        # Log the incoming message
        # logger.info(f"Received chat message for session {session_id}: {message[:50]}...")
        
        # Get Ollama service with the session-specific model if available, otherwise use default
        model_to_use = session_models.get(session_id, DEFAULT_LLM_MODEL)
        logger.info(f"Creating OllamaService with model: {model_to_use} for session: {session_id}")
        ollama_service = OllamaService(model=model_to_use)
        
        # Get context from session or initialize
        if session_id not in web_chat_sessions:
            logger.warning(f"Session {session_id} not found, creating new session")
            web_chat_sessions[session_id] = []
            
        context = web_chat_sessions[session_id]
        logger.info(f"Current context length for session {session_id}: {len(context)}")
        
        # Update context with user message
        user_message = {"role": "user", "content": message}
        context.append(user_message)
        
        # Store updated context immediately
        web_chat_sessions[session_id] = context
        logger.info(f"Updated context for session {session_id}, new length: {len(context)}")
        
        # Store the current message as the active message for this session
        # This ensures the streaming endpoint knows which message to process
        active_messages[session_id] = message
        logger.info(f"Set active message for session {session_id}: '{message[:50]}...'")
        
        # Start a task to store the user message in Milvus (don't wait for it to complete)
        # This allows the LLM to start processing immediately while the database write happens in the background
        async def store_user_message_async():
            try:
                embedding_service = EmbeddingService()
                user_embedding = embedding_service.get_embedding(message)
                
                # Store user message with sequence number based on context length
                sequence = len(context) - 1  # 0-indexed
                store_conversation_message(
                    session_id=session_id,
                    role="user",
                    content=message,
                    embedding=user_embedding,
                    sequence=sequence,
                    metadata={"model": DEFAULT_LLM_MODEL}
                )
            except Exception as e:
                logger.error(f"Error storing user message in background: {str(e)}", exc_info=e)
        
        # Start the storage task without awaiting it
        asyncio.create_task(store_user_message_async())
        
        # Store updated context
        web_chat_sessions[session_id] = context
        
        # Return a template that sets up the streaming connection
        return templates.TemplateResponse(
            "chat_stream.html",
            {
                "request": request,
                "session_id": session_id,
                "message": message
            }
        )
    except Exception as e:
        # In case of error, return error message
        error_message = {"role": "assistant", "content": f"An error occurred: {str(e)}"}
        
        # Get existing context or initialize
        context = web_chat_sessions.get(session_id, [])
        if context and len(context) > 0:
            # Add user message if not already there
            if context[-1].get("role") != "user":
                context.append({"role": "user", "content": message})
            # Add error message
            context.append(error_message)
        else:
            context = [{"role": "user", "content": message}, error_message]
            
        # Store updated context
        web_chat_sessions[session_id] = context
        
        return templates.TemplateResponse(
            "chat_messages.html", 
            {"request": request, "messages": context, "search_results": None}
        )


@router.get("/chat/models")
async def get_models_for_web(request: Request, session_id: Optional[str] = Query(None)):
    """
    Get available models for the web interface
    """
    try:
        # Get available models
        models = OllamaService.get_available_models()
        
        model_infos = []
        for model in models:
            model_info = {
                "name": model.get("name", ""),
                "size": model.get("size"),
                "modified_at": model.get("modified_at"),
                "digest": model.get("digest"),
                "is_current": (model.get("name") == DEFAULT_LLM_MODEL)
            }
            model_infos.append(model_info)
        
        # Get the current model for this session if available
        current_model = DEFAULT_LLM_MODEL
        if session_id and session_id in session_models:
            current_model = session_models[session_id]
            logger.info(f"Using session-specific model for session {session_id}: {current_model}")
        
        # Return JSON response
        return {
            "models": model_infos,
            "current_model": current_model,
            "cpu_model": CPU_LLM_MODEL,
            "gpu_model": GPU_LLM_MODEL
        }
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Error loading models: {str(e)}"}
        )


@router.get("/chat/models/switch")
async def switch_model_web(request: Request, model_name: str = Query(..., description="Model name to switch to"), session_id: Optional[str] = Query(None)):
    """
    Switch the active LLM model from the web interface
    """
    try:
        # Check if model exists
        models = OllamaService.get_available_models()
        model_names = [model.get("name") for model in models]
        
        if model_name not in model_names:
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": f"Model '{model_name}' not found"}
            )
        
        # Get session ID from query parameter or cookies
        if session_id:
            # Update the model for this specific session
            session_models[session_id] = model_name
            logger.info(f"Updated model for session {session_id} to {model_name}")
        else:
            # If no session ID provided, update the global default
            global DEFAULT_LLM_MODEL
            DEFAULT_LLM_MODEL = model_name
            logger.info(f"Updated global default model to {model_name}")
        
        return {"success": True, "message": f"Switched to model: {model_name}", "model": model_name}
    except Exception as e:
        logger.error(f"Error switching model: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error switching model: {str(e)}"}
        )
