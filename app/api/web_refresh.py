from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pathlib

from app.services.embedding import EmbeddingService
from app.database import search_documents
from app.api.web import web_chat_sessions

router = APIRouter(tags=["refresh"])

# Setup templates
BASE_DIR = pathlib.Path(__file__).parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Use the context manager service for session contexts
from app.services.context_manager import get_all_session_contexts, get_session_context

@router.get("/chat/refresh", response_class=HTMLResponse)
async def refresh_chat_message(
    request: Request,
    session_id: str = Query(...)
):
    """
    Get the formatted chat message with document context after streaming is complete
    """
    try:
        # Get context from session
        context = web_chat_sessions.get(session_id, [])
        
        if not context or len(context) < 2:
            return "<div class='p-4 bg-red-100 text-red-800 rounded-md'>No chat history found</div>"
        
        # Get the last user message and assistant response
        assistant_message = context[-1] if context[-1]["role"] == "assistant" else None
        
        if not assistant_message:
            return "<div class='p-4 bg-red-100 text-red-800 rounded-md'>No assistant response found</div>"
        
        # Find the corresponding user message
        user_message = None
        for i in range(len(context) - 2, -1, -1):
            if context[i]["role"] == "user":
                user_message = context[i]
                break
        
        if not user_message:
            return "<div class='p-4 bg-red-100 text-red-800 rounded-md'>No user message found</div>"
        
        # Get stored context information for this session
        session_context = get_session_context(session_id)
        
        # Retrieve document and conversation context
        document_context = session_context.get('document_context', [])
        conversation_context = session_context.get('conversation_context', [])
        
        # Use search_results if available or combine document and conversation contexts
        relevant_results = session_context.get('search_results', [])
        
        # If no search_results, but we have document_context, use that
        if not relevant_results and document_context:
            relevant_results = document_context
            
        # Determine if we should show context
        show_context = len(relevant_results) > 0
        
        # Return HTML response with the assistant message and document context
        return templates.TemplateResponse(
            "chat_message.html",
            {
                "request": request,
                "message": assistant_message["content"],
                "is_assistant": True,
                "search_results": relevant_results if show_context else None,
                "document_context": document_context,
                "conversation_context": conversation_context,
                "message_with_context": show_context
            }
        )
    except Exception as e:
        return f"<div class='chat-message chat-assistant'><p>Error refreshing message: {str(e)}</p></div>"

@router.get("/chat/context")
async def get_chat_context(request: Request, session_id: str = Query(...)):
    """
    🧠 **Neural Context Inspection**: Retrieve the knowledge fragments that informed AI responses
    
    Access the underlying neural pathways and knowledge fragments that influenced
    the most recent thought synthesis, enabling transparency into the cognitive process.
    """
    # Get context for this session using the context manager service
    context = get_session_context(session_id)
    
    # Add logging to debug content
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Context for session {session_id}: {len(context.get('document_context', []))} documents, {len(context.get('conversation_context', []))} conversations")
    
    # Ensure document and conversation contexts are lists
    if 'document_context' not in context or not isinstance(context['document_context'], list):
        context['document_context'] = []
    
    if 'conversation_context' not in context or not isinstance(context['conversation_context'], list):
        context['conversation_context'] = []
    
    # If we have search_results but no specific contexts, separate them by type
    if 'search_results' in context and (not context['document_context'] and not context['conversation_context']):
        document_results = []
        conversation_results = []
        
        for result in context['search_results']:
            if result.get('type') == 'document' or ('file_name' in result and 'session_id' not in result):
                # Make sure type is explicitly set
                result['type'] = 'document'
                document_results.append(result)
            elif result.get('type') == 'conversation' or ('session_id' in result and 'role' in result):
                # Make sure type is explicitly set
                result['type'] = 'conversation'
                conversation_results.append(result)
        
        context['document_context'] = document_results
        context['conversation_context'] = conversation_results
    
    return JSONResponse(content=context)
