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
            
            # Search in vector database for documents
            try:
                doc_search_hits = search_documents(query_embedding, 3)  # Get top 3 most relevant document chunks
                logger.info(f"Document search returned {len(doc_search_hits)} result sets")
                
                # Add more detailed logging about the structure of doc_search_hits
                logger.info(f"Doc search hits type: {type(doc_search_hits)}")
                if len(doc_search_hits) > 0:
                    # Log first result structure
                    first_hits = doc_search_hits[0]
                    logger.info(f"First hits type: {type(first_hits)}, content structure: {first_hits}")
                    
                    # If first_hits is a list and has items
                    if hasattr(first_hits, '__iter__') and len(first_hits) > 0:
                        first_hit = first_hits[0]
                        logger.info(f"First hit type: {type(first_hit)}, dir: {dir(first_hit)}")
                        # Try to access common Milvus search result attributes
                        if hasattr(first_hit, 'entity'):
                            logger.info(f"Entity type: {type(first_hit.entity)}, dir: {dir(first_hit.entity)}")
            except Exception as doc_e:
                logger.error(f"Error in document search: {str(doc_e)}")
                doc_search_hits = []
            
            # Search in conversation history
            try:
                from app.database import search_conversations
                conv_search_hits = search_conversations(query_embedding, 3)  # Get top 3 most relevant conversation chunks
                logger.info(f"Conversation search returned {len(conv_search_hits)} result sets")
            except Exception as conv_e:
                logger.error(f"Error in conversation search: {str(conv_e)}")
                conv_search_hits = []
            
            # Format search results and filter by relevance
            relevant_results = []
            min_relevance_threshold = 0.6  # Minimum relevance score to consider a result relevant
            
            # Process document results
            for hits in doc_search_hits:
                try:
                    # Log the hits data structure for debugging
                    logger.info(f"Document hits type: {type(hits)}, content: {hits}")
                    
                    # Handle different potential data structures
                    # Case 1: hits is already a list
                    if isinstance(hits, list):
                        hit_list = hits
                    # Case 2: hits is a single object
                    else:
                        hit_list = [hits]
                    
                    for hit in hit_list:
                        try:
                            # Create a simple dictionary with the necessary fields
                            result = {
                                "file_id": "",  # Will be populated if available
                                "file_name": "",  # Will be populated if available
                                "content": "",  # Will be populated if available
                                "chunk_id": "",  # Will be populated if available
                                "score": 0.0,  # Default score
                                "type": "document"  # Document type
                            }
                            
                            # Log the hit object for debugging
                            logger.info(f"Processing document hit: {hit}")
                            
                            # Try to get the score - this field is critical for relevance ranking
                            # For Milvus results, the score is directly accessible as hit.score
                            if hasattr(hit, 'score'):
                                result["score"] = float(hit.score)
                                logger.info(f"Found score via attribute: {result['score']}")
                            # For dict-like objects
                            elif isinstance(hit, dict) and 'score' in hit:
                                result["score"] = float(hit['score'])
                                logger.info(f"Found score via dictionary: {result['score']}")
                            # Sometimes the distance/score is in a distance field
                            elif hasattr(hit, 'distance'):
                                result["score"] = float(hit.distance)
                                logger.info(f"Found score via distance attribute: {result['score']}")
                            # Try to access score in case object has unusual structure
                            else:
                                try:
                                    # Try to get any attribute that might contain score
                                    for attr_name in dir(hit):
                                        if 'score' in attr_name.lower() or 'distance' in attr_name.lower() or 'similarity' in attr_name.lower():
                                            if not attr_name.startswith('__') and callable(getattr(hit, attr_name)) == False:
                                                val = getattr(hit, attr_name)
                                                if isinstance(val, (int, float)):
                                                    result["score"] = float(val)
                                                    logger.info(f"Found score via dynamic attribute {attr_name}: {result['score']}")
                                                    break
                                    
                                    # If we still don't have a score but we're processing Milvus results
                                    # The first hit should always have the highest score - give it a high score
                                    if result["score"] == 0.0 and len(doc_search_hits) > 0:
                                        result["score"] = 0.95  # Give a high score to ensure it passes threshold
                                        logger.warning(f"Could not extract score from hit, using default high score: {result['score']}")
                                except Exception as score_e:
                                    logger.error(f"Error finding score field: {str(score_e)}")
                            
                            # Try different ways to access entity data
                            # Case 1: hit has an entity attribute - Milvus search results format
                            if hasattr(hit, 'entity'):
                                entity = hit.entity
                                logger.info(f"Processing entity: {entity}")
                                
                                # Case 1.1: Try entity.get() method first (standard Milvus way)
                                if hasattr(entity, 'get') and callable(entity.get):
                                    try:
                                        # Try to get values using get method
                                        for key in ['file_id', 'file_name', 'content', 'chunk_id']:
                                            try:
                                                # Milvus Python SDK might return values in different ways
                                                # Try multiple approaches
                                                
                                                # Approach 1: Direct get method
                                                val = entity.get(key)
                                                if val is not None:
                                                    result[key] = val
                                                    logger.info(f"Got {key} using entity.get(): {val[:30] if isinstance(val, str) else val}")
                                                    continue  # Skip to next key if we got this one
                                                
                                                # Approach 2: Accessing fields property
                                                if hasattr(entity, 'fields') and isinstance(entity.fields, dict) and key in entity.fields:
                                                    result[key] = entity.fields[key]
                                                    logger.info(f"Got {key} using entity.fields: {result[key][:30] if isinstance(result[key], str) else result[key]}")
                                                    continue
                                                    
                                                # Approach 3: Using __getitem__ if available
                                                if hasattr(entity, '__getitem__'):
                                                    try:
                                                        val = entity[key]
                                                        if val is not None:
                                                            result[key] = val
                                                            logger.info(f"Got {key} using entity[key]: {val[:30] if isinstance(val, str) else val}")
                                                            continue
                                                    except (KeyError, IndexError, TypeError):
                                                        pass
                                                        
                                            except Exception as key_e:
                                                logger.warning(f"Error getting {key} using entity.get() method: {str(key_e)}")
                                    except Exception as get_e:
                                        logger.warning(f"Error using entity.get(): {str(get_e)}")
                                
                                # Case 1.2: entity has __dict__
                                if hasattr(entity, '__dict__'):
                                    entity_dict = entity.__dict__
                                    for key in ['file_id', 'file_name', 'content', 'chunk_id']:
                                        if key in entity_dict:
                                            result[key] = entity_dict[key]
                                            logger.info(f"Got {key} using entity.__dict__: {result[key][:30] if isinstance(result[key], str) else result[key]}")
                                
                                # Case 1.3: Direct attribute access
                                if not result['file_id'] and hasattr(entity, 'file_id'):
                                    result['file_id'] = entity.file_id
                                    logger.info(f"Got file_id using direct access: {result['file_id']}")
                                if not result['file_name'] and hasattr(entity, 'file_name'):
                                    result['file_name'] = entity.file_name
                                    logger.info(f"Got file_name using direct access: {result['file_name']}")
                                if not result['content'] and hasattr(entity, 'content'):
                                    result['content'] = entity.content
                                    logger.info(f"Got content using direct access: {result['content'][:30]}...")
                                if not result['chunk_id'] and hasattr(entity, 'chunk_id'):
                                    result['chunk_id'] = entity.chunk_id
                                    logger.info(f"Got chunk_id using direct access: {result['chunk_id']}")
                            
                            # Case 2: hit is a dictionary
                            elif isinstance(hit, dict):
                                logger.info(f"Hit is a dictionary with keys: {hit.keys()}")
                                for key in ['file_id', 'file_name', 'content', 'chunk_id']:
                                    if key in hit:
                                        result[key] = hit[key]
                                        logger.info(f"Got {key} from hit dict: {result[key][:30] if isinstance(result[key], str) else result[key]}")
                                
                                # Case 2.1: hit has an entity key that is a dict
                                if 'entity' in hit and isinstance(hit['entity'], dict):
                                    entity = hit['entity']
                                    logger.info(f"Found entity dict with keys: {entity.keys()}")
                                    for key in ['file_id', 'file_name', 'content', 'chunk_id']:
                                        if key in entity:
                                            result[key] = entity[key]
                                            logger.info(f"Got {key} from entity dict: {result[key][:30] if isinstance(result[key], str) else result[key]}")
                            
                            # Before finalizing the result, ensure we have fallback values for critical fields
                            # These fallbacks ensure that even if extraction fails, we have usable information
                            if not result['file_name'] or result['file_name'] == "":
                                result['file_name'] = "Document"
                                logger.warning(f"Using fallback filename: {result['file_name']}")
                            
                            if not result['content'] or result['content'] == "":
                                # Try to extract content from the raw hit object
                                try:
                                    # Dump the entire hit object to string and extract any text that might be content
                                    hit_str = str(hit)
                                    logger.info(f"Searching for content in raw hit object: {hit_str[:100]}...")
                                    
                                    # Look for content patterns in the raw string representation
                                    content_patterns = [
                                        "content='([^']+)'",  # content='text'
                                        'content":"([^"]+)"',  # content":"text"
                                        "'content': '([^']+)'",  # 'content': 'text'
                                        '"content":"([^"]+)"',  # "content":"text"
                                    ]
                                    
                                    import re
                                    for pattern in content_patterns:
                                        match = re.search(pattern, hit_str)
                                        if match:
                                            content = match.group(1)
                                            if len(content) > 10:  # Ensure it's meaningful content
                                                result['content'] = content
                                                logger.info(f"Extracted content from raw hit using regex: {content[:50]}...")
                                                break
                                except Exception as extract_e:
                                    logger.error(f"Error extracting content from raw hit: {str(extract_e)}")
                            
                            if not result['content'] or result['content'] == "":
                                result['content'] = "No content available. The document was found but its content could not be extracted."
                                logger.warning(f"Using fallback content: {result['content']}")
                            
                            # Add the result regardless of the relevance threshold for debugging
                            # We'll log it but only append if it meets the threshold
                            logger.info(f"Document result prepared: {result}")
                            
                            # Only include results that meet the relevance threshold
                            if result["score"] >= min_relevance_threshold:
                                relevant_results.append(result)
                                logger.info(f"Added document result to relevant_results, now has {len(relevant_results)} items")
                            else:
                                logger.info(f"Document result score {result['score']} below threshold {min_relevance_threshold}, not adding")
                        except Exception as hit_e:
                            logger.error(f"Error processing document hit: {str(hit_e)}", exc_info=True)
                except Exception as hits_e:
                    logger.error(f"Error processing document hits: {str(hits_e)}", exc_info=True)
            
            # Process conversation results
            for hits in conv_search_hits:
                try:
                    for hit in hits:
                        try:
                            # Create a simple dictionary with the necessary fields
                            # Avoid using hit.entity.get() which is causing problems
                            result = {
                                "session_id": "",  # Will be populated if available
                                "role": "",  # Will be populated if available
                                "content": "",  # Will be populated if available
                                "timestamp": "",  # Will be populated if available
                                "chunk_id": "",  # Will be populated if available
                                "score": 0.0,  # Default score
                                "type": "conversation"  # Conversation type
                            }
                            
                            # Apply the same improved score extraction logic for conversation results
                            if hasattr(hit, 'score'):
                                result["score"] = float(hit.score)
                                logger.info(f"Found conversation score via attribute: {result['score']}")
                            # For dict-like objects
                            elif isinstance(hit, dict) and 'score' in hit:
                                result["score"] = float(hit['score'])
                                logger.info(f"Found conversation score via dictionary: {result['score']}")
                            # Sometimes the distance/score is in a distance field
                            elif hasattr(hit, 'distance'):
                                result["score"] = float(hit.distance)
                                logger.info(f"Found conversation score via distance attribute: {result['score']}")
                            # Try to access score in case object has unusual structure
                            else:
                                try:
                                    # Try to get any attribute that might contain score
                                    for attr_name in dir(hit):
                                        if 'score' in attr_name.lower() or 'distance' in attr_name.lower() or 'similarity' in attr_name.lower():
                                            if not attr_name.startswith('__') and callable(getattr(hit, attr_name)) == False:
                                                val = getattr(hit, attr_name)
                                                if isinstance(val, (int, float)):
                                                    result["score"] = float(val)
                                                    logger.info(f"Found conversation score via dynamic attribute {attr_name}: {result['score']}")
                                                    break
                                    
                                    # If we still don't have a score but we're processing Milvus results
                                    if result["score"] == 0.0 and len(conv_search_hits) > 0:
                                        result["score"] = 0.95  # Give a high score to ensure it passes threshold
                                        logger.warning(f"Could not extract conversation score from hit, using default high score: {result['score']}")
                                except Exception as score_e:
                                    logger.error(f"Error finding conversation score field: {str(score_e)}")
                            
                            # Try to access entity fields directly using __dict__ or vars
                            if hasattr(hit, 'entity'):
                                entity = hit.entity
                                
                                # Try different ways to access entity data
                                if hasattr(entity, '__dict__'):
                                    entity_dict = entity.__dict__
                                    for key in ['session_id', 'role', 'content', 'timestamp', 'chunk_id']:
                                        if key in entity_dict:
                                            result[key] = entity_dict[key]
                                
                                # If the above didn't work, try direct attribute access
                                if not result['session_id'] and hasattr(entity, 'session_id'):
                                    result['session_id'] = entity.session_id
                                if not result['role'] and hasattr(entity, 'role'):
                                    result['role'] = entity.role
                                if not result['content'] and hasattr(entity, 'content'):
                                    result['content'] = entity.content
                                if not result['timestamp'] and hasattr(entity, 'timestamp'):
                                    result['timestamp'] = entity.timestamp
                                if not result['chunk_id'] and hasattr(entity, 'chunk_id'):
                                    result['chunk_id'] = entity.chunk_id
                            
                            # Only include results that meet the relevance threshold
                            if result["score"] >= min_relevance_threshold:
                                relevant_results.append(result)
                        except Exception as hit_e:
                            logger.error(f"Error processing conversation hit: {str(hit_e)}")
                except Exception as hits_e:
                    logger.error(f"Error processing conversation hits: {str(hits_e)}")
            
            # Sort all results by relevance score
            relevant_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Determine if we should include context based on the query and results
            include_context = len(relevant_results) > 0
            
            # Get streaming response from LLM
            if include_context and relevant_results:
                # Get response with context from both documents and conversations
                doc_count = len([r for r in relevant_results if r.get('type') == 'document'])
                conv_count = len([r for r in relevant_results if r.get('type') == 'conversation'])
                logger.info(f"Using search_and_respond with {len(relevant_results)} relevant chunks ({doc_count} documents, {conv_count} conversations)")
                logger.info(f"Top result relevance: {relevant_results[0]['score'] if relevant_results else 'N/A'}, type: {relevant_results[0].get('type') if relevant_results else 'N/A'}")
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
                    
                    # Get the user message that this is a response to
                    user_message_data = None
                    if user_message and isinstance(user_message, dict):
                        user_message_data = {
                            "role": user_message.get("role", "user"),
                            "content": user_message.get("content", ""),
                            "sequence": len(context) - 2  # The user message is the second-to-last in context
                        }
                    
                    # Store the assistant message with reference to the user message
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
                        },
                        related_message=user_message_data
                    )
                    
                    # Now update the user message to include a reference to this assistant response
                    # This creates a bidirectional link between the messages
                    if user_message_data:
                        try:
                            # Create a reference to the assistant message that we just stored
                            assistant_message_data = {
                                "role": "assistant",
                                "content": full_response,
                                "sequence": sequence
                            }
                            
                            # Get user message's actual content and embedding
                            user_content = user_message.get("content", "")
                            
                            # We need to re-embed the user message to store it again
                            try:
                                user_embedding = embedding_service.get_embedding(user_content)
                                
                                # Re-store the user message with a reference to the assistant response
                                # We're effectively updating the user message in the database
                                store_conversation_message(
                                    session_id=session_id,
                                    role="user",
                                    content=user_content,
                                    embedding=user_embedding,
                                    sequence=user_message_data["sequence"],
                                    metadata={
                                        "model": ollama_service.model,
                                        "updated": True,  # Mark this as an update
                                        "update_reason": "Added assistant response reference"
                                    },
                                    related_message=assistant_message_data
                                )
                                logger.info(f"Updated user message with reference to assistant response for session {session_id}")
                            except Exception as embed_e:
                                logger.error(f"Error re-embedding user message: {str(embed_e)}", exc_info=embed_e)
                        except Exception as update_e:
                            logger.error(f"Error updating user message with assistant reference: {str(update_e)}", exc_info=update_e)
                except Exception as e:
                    logger.error(f"Error storing message in background: {str(e)}", exc_info=e)
            
            # Start the storage task without awaiting it
            asyncio.create_task(store_message_async())
            
            # Store the search results in the context manager for display in the UI
            from app.services.context_manager import set_session_context
            
            # Store context information
            # Separate document and conversation results
            document_results = [r for r in relevant_results if r.get('type') == 'document']
            
            # Filter out conversation results from the current session
            # This ensures the context sources overlay doesn't show messages from the current conversation
            conversation_results = [r for r in relevant_results if r.get('type') == 'conversation' 
                                   and r.get('session_id') != session_id]
            
            # Combine filtered results
            filtered_results = document_results + conversation_results
            
            # Store the context information - only store filtered search_results
            # This prevents duplication and removes current conversation messages from the context overlay
            set_session_context(session_id, {
                "search_results": filtered_results
            })
            
            # Signal that streaming is complete with context information
            has_context = len(relevant_results) > 0
            yield f"data: {{\"done\": true, \"has_context\": {str(has_context).lower()}}}\n\n"
            
            # Log completion with detailed context information
            doc_count = len([r for r in relevant_results if r.get('type') == 'document'])
            conv_count = len([r for r in relevant_results if r.get('type') == 'conversation'])
            logger.info(f"Streaming complete for session {session_id}. Has context: {has_context}, Document results: {doc_count}, Conversation results: {conv_count}")
            
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
