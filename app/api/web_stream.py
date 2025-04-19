from fastapi import APIRouter, Request, Query
from fastapi.responses import StreamingResponse
import json
import asyncio
from typing import Dict, Any, Optional

from app.services.embedding import EmbeddingService
from app.services.ollama import OllamaService
from app.database import search_documents, store_conversation_message
from app.config import DEFAULT_LLM_MODEL
from app.api.web import session_models
from app.utils.logging import get_logger

# Import the web chat sessions and active messages dictionaries
from app.api.web import web_chat_sessions, active_messages

router = APIRouter(tags=["streaming"])

# Setup logging
logger = get_logger(__name__)

@router.get("/chat/stream")
async def stream_chat_response(request: Request, session_id: str = Query(...), message_param: Optional[str] = Query(None, alias="message")):
    """
    Stream the LLM response for a chat message
    """
    logger.info(f"Starting streaming response for session {session_id}")
    
    async def event_generator():
        try:
            # Get Ollama service with the session-specific model if available, otherwise use default
            model_to_use = session_models.get(session_id, DEFAULT_LLM_MODEL)
            logger.info(f"Creating OllamaService with model: {model_to_use} for session: {session_id}")
            ollama_service = OllamaService(model=model_to_use)
            
            # Get context from session
            if session_id not in web_chat_sessions:
                logger.error(f"Session ID {session_id} not found in web_chat_sessions")
                yield f"data: {json.dumps({'error': 'Session not found. Please refresh the page and try again.'})}\n\n"
                yield "data: {\"done\": true}\n\n"
                return
                
            context = web_chat_sessions[session_id]
            logger.info(f"Retrieved context for session {session_id}, length: {len(context)}")
            
            # Check if we have an active message for this session
            active_message = None
            
            # First check if message was passed directly in the URL
            if message_param:
                logger.info(f"Using message from URL parameter: '{message_param[:50]}...'")
                active_message = message_param
                
                # Ensure this message is in the context
                message_in_context = False
                for msg in context:
                    if msg.get("role") == "user" and msg.get("content") == message_param:
                        message_in_context = True
                        break
                        
                if not message_in_context:
                    logger.warning(f"Message from URL not found in context, adding it now")
                    user_message = {"role": "user", "content": message_param}
                    context.append(user_message)
                    web_chat_sessions[session_id] = context
            
            # If no message in URL, check active_messages
            elif session_id in active_messages:
                active_message = active_messages.get(session_id)
                logger.info(f"Found active message for session {session_id}: '{active_message[:50]}...'")
            
            # Verify we have a valid context with at least one user message
            if not context or len(context) < 1:
                if active_message:
                    # We have an active message but no context - create context with the active message
                    logger.warning(f"Empty context for session {session_id} but found active message, creating context")
                    user_message = {"role": "user", "content": active_message}
                    context.append(user_message)
                    web_chat_sessions[session_id] = context
                    logger.info(f"Added active message to context for session {session_id}")
                else:
                    # No context and no active message - use a default greeting
                    logger.warning(f"Empty context for session {session_id}, creating a new context with a default message")
                    user_message = {"role": "user", "content": "Hello, I'd like to chat about my documents."}
                    context.append(user_message)
                    web_chat_sessions[session_id] = context
                    logger.info(f"Added default message to empty context for session {session_id}")
            
            # Get the last user message
            user_message = None
            
            # First check if we have an active message that should be processed
            if active_message:
                # Find this message in the context
                for msg in context:
                    if msg.get("role") == "user" and msg.get("content") == active_message:
                        user_message = msg
                        logger.info(f"Using active message from context: '{active_message[:50]}...'")
                        # If it was from active_messages, remove it after processing
                        if session_id in active_messages:
                            del active_messages[session_id]
                        break
            
            # If no active message found, use the last user message
            if not user_message:
                for msg in reversed(context):
                    if msg.get("role") == "user":
                        user_message = msg
                        logger.info(f"Using last user message from context: '{msg.get('content', '')[:50]}...'")
                        break
            
            if not user_message:
                # No user message found
                logger.error(f"No user message found in context for session {session_id}")
                yield f"data: {json.dumps({'error': 'No user message found. Please refresh the page and try again.'})}\n\n"
                yield "data: {\"done\": true}\n\n"
                return
                
            logger.info(f"Found user message: {user_message['content'][:50]}...")
            
            # Log the full context being sent to the LLM
            logger.info(f"Full context for session {session_id}: {context}")
            
            # Get message content directly from the user message
            message = user_message.get("content", "")
            logger.info(f"Processing user query for session {session_id}: '{message}'")
            
            # Create a clean context for the LLM that only includes previous messages
            clean_context = []
            for i, msg in enumerate(context):
                # Only include messages before the current one
                if i < len(context) - 1:  # Skip the last message (current user message)
                    clean_context.append(msg)
            
            # Log the clean context
            logger.info(f"Clean context for LLM (excluding current message): {json.dumps(clean_context, indent=2)}")
            logger.info(f"Current user message being processed: {json.dumps(user_message, indent=2)}")
            
            # Generate embedding for search
            embedding_service = EmbeddingService()
            query_embedding = embedding_service.get_embedding(message)
            
            # Search in vector database
            search_hits = search_documents(query_embedding, 5)  # Get top 5 most relevant chunks
            
            # Format search results and filter by relevance
            relevant_results = []
            min_relevance_threshold = 0.6  # Minimum relevance score to consider a document relevant
            
            for hits in search_hits:
                for hit in hits:
                    score = float(hit.score)
                    result = {
                        "file_id": hit.entity.get('file_id'),
                        "file_name": hit.entity.get('file_name'),
                        "content": hit.entity.get('content'),
                        "chunk_id": hit.entity.get('chunk_id'),
                        "score": score
                    }
                    
                    # Only include results that meet the relevance threshold
                    if score >= min_relevance_threshold:
                        relevant_results.append(result)
            
            # Determine if we should include context based on the query and results
            include_context = len(relevant_results) > 0
            
            # Get streaming response from LLM
            if include_context and relevant_results:
                # Get response with document context
                logger.info(f"Using search_and_respond with {len(relevant_results)} relevant document chunks")
                logger.info(f"Top document relevance: {relevant_results[0]['score'] if relevant_results else 'N/A'}")
                response_stream = ollama_service.search_and_respond(
                    query=message,
                    search_results=relevant_results,
                    context=clean_context,  # Use clean context (previous messages only)
                    stream=True
                )
            else:
                # No relevant documents found, just use the regular chat response
                logger.info(f"No relevant documents found, using regular chat response")
                response_stream = ollama_service.generate_response(
                    prompt=message,
                    context=clean_context,  # Use clean context (previous messages only)
                    system_prompt="You are a helpful AI assistant. Be concise and clear in your responses.",
                    stream=True
                )
            
            # Variables to accumulate the full response
            full_response = ""
            
            # Stream the response chunks
            for chunk in response_stream:
                if "error" in chunk:
                    # Handle error
                    error_msg = chunk.get("error", "Unknown error")
                    logger.error(f"Error in LLM response: {error_msg}")
                    yield f"data: {json.dumps({'content': f'Error: {error_msg}'})}\n\n"
                    break
                
                # Extract content from the chunk
                content = ""
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                elif "response" in chunk:
                    content = chunk["response"]
                
                # Debug logging
                # logger.info(f"Received chunk: {chunk}")
                # logger.info(f"Extracted content: {content}")
                
                if content:
                    # Send the content chunk
                    full_response += content
                    yield f"data: {json.dumps({'content': content})}\n\n"
                    
                    # Small delay to avoid overwhelming the client
                    await asyncio.sleep(0.01)
            
            # Create assistant message
            assistant_message = {"role": "assistant", "content": full_response}
            
            # Add to context
            context.append(assistant_message)
            
            # Store updated context
            web_chat_sessions[session_id] = context
            
            # Start a task to store the assistant message in Milvus (don't wait for it to complete)
            # This allows the streaming response to complete quickly while the database write happens in the background
            async def store_message_async():
                try:
                    assistant_embedding = embedding_service.get_embedding(full_response)
                    sequence = len(context) - 1  # 0-indexed
                    store_conversation_message(
                        session_id=session_id,
                        role="assistant",
                        content=full_response,
                        embedding=assistant_embedding,
                        sequence=sequence,
                        metadata={
                            "model": ollama_service.model,
                            "has_context": include_context,
                            "context_count": len(relevant_results) if include_context else 0
                        }
                    )
                except Exception as e:
                    logger.error(f"Error storing message in background: {str(e)}", exc_info=e)
            
            # Start the storage task without awaiting it
            asyncio.create_task(store_message_async())
            
            # Signal that streaming is complete with context information
            has_context = len(relevant_results) > 0
            yield f"data: {{\"done\": true, \"has_context\": {str(has_context).lower()}}}\n\n"
            
            # Log completion
            logger.info(f"Streaming complete for session {session_id}. Context: {has_context}")
            
        except Exception as e:
            # Handle any exceptions
            error_msg = str(e)
            logger.error(f"Streaming error: {error_msg}", exc_info=e)
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            yield "data: {\"done\": true, \"has_context\": false}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
