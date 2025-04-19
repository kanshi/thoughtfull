from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pathlib

from app.services.embedding import EmbeddingService
from app.database import search_documents
from app.api.web import web_chat_sessions

router = APIRouter(tags=["refresh"])

# Setup templates
BASE_DIR = pathlib.Path(__file__).parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

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
        
        # Generate embedding for search
        embedding_service = EmbeddingService()
        query_embedding = embedding_service.get_embedding(user_message["content"])
        
        # Search in vector database
        search_hits = search_documents(query_embedding, 5)
        
        # Format search results and filter by relevance
        relevant_results = []
        min_relevance_threshold = 0.6
        
        for hits in search_hits:
            for hit in hits:
                score = float(hit.score)
                if score >= min_relevance_threshold:
                    relevant_results.append({
                        "file_id": hit.entity.get('file_id'),
                        "file_name": hit.entity.get('file_name'),
                        "content": hit.entity.get('content'),
                        "chunk_id": hit.entity.get('chunk_id'),
                        "score": score
                    })
        
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
                "message_with_context": show_context
            }
        )
    except Exception as e:
        return f"<div class='chat-message chat-assistant'><p>Error refreshing message: {str(e)}</p></div>"
