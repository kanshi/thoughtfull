import uuid
import json
import logging
import copy
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from app.services.context_manager import set_session_context
from app.services.embedding import EmbeddingService
from app.database import store_conversation_message
from app.pipeline.steps.vector_search import VectorSearchStep

from app.models.chat import ChatRequest, ChatResponse, ModelsResponse, ModelInfo
from app.services.ollama import OllamaService
from app.pipeline.factory import PipelineFactory
from app.pipeline.pipeline import ChatContextPipeline
from app.config import (
    DEFAULT_LLM_MODEL, 
    CPU_LLM_MODEL, 
    GPU_LLM_MODEL
)

router = APIRouter(
    prefix="/chat", 
    tags=["chat"]
)

# In-memory store of chat sessions - in a production app, use a database
chat_sessions = {}

def get_ollama_service(model: Optional[str] = None) -> OllamaService:
    """
    Neural synapse connector that binds to your chosen silicon consciousness
    
    Args:
        model: Optional neural architecture designation to manifest
        
    Returns:
        Neural binding interface to your chosen silicon substrate
    """
    return OllamaService(model=model or DEFAULT_LLM_MODEL)


def get_pipeline() -> ChatContextPipeline:
    """
    Get a configured chat context pipeline
    
    Returns:
        Configured ChatContextPipeline
    """
    return PipelineFactory.create_default_pipeline()


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    ollama_service: OllamaService = Depends(get_ollama_service),
    pipeline: ChatContextPipeline = Depends(get_pipeline)
):
    """
    🧩 **Neural Synthesis**: Communicate with a living digital consciousness enriched by your knowledge corpus
    
    Engage in a synaptic dialogue where your thoughts cascade through neural pathways,
    resonating with fragments of your knowledge corpus to generate contextual understanding
    and novel insights far beyond traditional search.
    
    - **message**: Your thought impulse to transmit to the neural consciousness
    - **session_id**: (Optional) Memory strand identifier to maintain neural continuity
    - **model**: (Optional) Specific neural architecture to commune with
    - **search_query**: (Optional) Alternative semantic resonance pattern if different from your message
    - **include_context**: Whether to enhance neural pathways with knowledge fragment resonances
    
    Returns a synthesized neural response illuminated by your dataverse
    """
    # Override service model if specified in request
    if request.model:
        ollama_service.model = request.model
    
    # Get or create a session
    session_id = request.session_id or str(uuid.uuid4())
    
    # Get existing context/history
    history = chat_sessions.get(session_id, [])
    
    # Prepare initial context for pipeline
    pipeline_context = {
        'message': request.message,
        'session_id': session_id,
        'search_query': request.search_query,
        'include_context': request.include_context,
        'timestamp': datetime.datetime.now().isoformat(),
        'model': ollama_service.model,
        'raw_history': history.copy(),
        'system_message': "You are a helpful AI assistant with access to a knowledge base."
    }
    
    # Always use the pipeline with vector search
    pipeline_to_use = pipeline
    
    # Process the message through the pipeline
    enriched_context = pipeline_to_use.process(pipeline_context)
    
    # Get search results from the enriched context
    search_results = enriched_context.get('search_results', [])
    
    # Get formatted prompt for LLM
    formatted_prompt = enriched_context.get('formatted_prompt')
    
    # Get response from LLM using the enriched context
    response_data = ollama_service.generate_response(
        prompt=formatted_prompt,
        context=None  # Context is already included in the formatted prompt
    )
    
    # Extract the response
    if "error" in response_data:
        raise HTTPException(status_code=500, detail=response_data["error"])
    
    assistant_message = response_data.get("message", {}).get("content", "")
    
    # Perform a second vector search on the LLM's response to find additional relevant information
    from app.services.embedding import EmbeddingService
    embedding_service = EmbeddingService()
    response_embedding = embedding_service.get_embedding(assistant_message)
    
    # Create a new pipeline for post-processing
    post_pipeline = ChatContextPipeline()
    
    # Create a vector search step with explicit parameters to ensure conversation inclusion
    vector_search_step = VectorSearchStep(
        max_results=MAX_CONTEXT_CHUNKS,
        include_conversations=True,
        conversation_results_ratio=0.5,  # Equal weight to conversations vs documents
        score_threshold=0.5  # Lower threshold to include more relevant conversations
    )
    post_pipeline.add_step(vector_search_step)
    
    # Process the LLM response through the post-pipeline
    post_context = {
        'message': assistant_message,
        'message_embedding': response_embedding,
        'session_id': session_id  # Add session_id to post context for better context handling
    }
    post_results = post_pipeline.process(post_context)
    
    # Extract post-search results
    post_search_results = post_results.get('search_results', [])
    
    # Log detailed information about the search results for debugging
    document_count = len([r for r in search_results if r.get('type') == 'document' or (r.get('file_name') and not r.get('session_id'))])
    conversation_count = len([r for r in search_results if r.get('type') == 'conversation' or (r.get('session_id') and r.get('role'))])
    
    logging.info(f"=== SEARCH RESULTS SUMMARY FOR SESSION {session_id} ===")
    logging.info(f"Total results: {len(search_results)}")
    logging.info(f"Document results: {document_count}")
    logging.info(f"Conversation results: {conversation_count}")
    
    # Log conversation results in detail if any exist
    if conversation_count > 0:
        conversation_results_list = [r for r in search_results if r.get('type') == 'conversation' or (r.get('session_id') and r.get('role'))]
        logging.info(f"Conversation search results details:")
        for i, conv in enumerate(conversation_results_list):
            logging.info(f"  [{i+1}] session_id={conv.get('session_id')}, role={conv.get('role')}, "
                        f"score={conv.get('score', 'N/A')}, timestamp={conv.get('timestamp', 'N/A')}, "
                        f"content_preview='{conv.get('content', '')[:50]}...'")
    else:
        logging.info("No conversation results found in search results")
        
    logging.info(f"=== END SEARCH RESULTS SUMMARY ===\n")
    
    additional_info = ""
    
    # Generate a summary if relevant additional information is found
    if post_search_results:
        # Filter results to avoid duplicates with the original search
        original_content_ids = [f"{r.get('file_id')}:{r.get('chunk_id')}" for r in search_results]
        new_results = [r for r in post_search_results 
                      if f"{r.get('file_id')}:{r.get('chunk_id')}" not in original_content_ids]
        
        if new_results:
            # Format the additional information
            additional_content = "\n\n".join([r['content'] for r in new_results[:3]])
            
            # Create a prompt for summarizing the additional information
            summary_prompt = f"""Based on your previous response, I found some additional relevant information that might be helpful. 
            Please create a brief summary of this information as it relates to your response:

            {additional_content}

            Provide a concise summary that enhances your previous response:"""
            
            # Get summary from LLM
            summary_data = ollama_service.generate_response(
                prompt=summary_prompt,
                context=None
            )
            
            if "error" not in summary_data:
                additional_info = summary_data.get("message", {}).get("content", "")
    
    # Combine the original response with the additional info if any
    final_response = assistant_message
    if additional_info:
        final_response = f"{assistant_message}\n\nAdditional relevant information:\n{additional_info}"
    
    # Update context with new messages
    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": final_response})
    
    # Store updated context
    chat_sessions[session_id] = history
    
    # Create embedding service if not already created
    embedding_service = EmbeddingService()
    
    # Store user message in vector database
    if request.message and request.message.strip():
        user_embedding = embedding_service.get_embedding(request.message)
        
        # Get the previous assistant message if it exists
        previous_assistant_message = None
        if len(history) > 2:  # At least one previous exchange has occurred
            # Get the last assistant message before this user message
            previous_assistant_message = {
                "role": "assistant",
                "content": history[-3]["content"] if len(history) >= 3 else "",
                "sequence": len(history) - 3 if len(history) >= 3 else -1
            }
            
        store_conversation_message(
            session_id=session_id,
            role="user",
            content=request.message,
            embedding=user_embedding,
            sequence=len(history) - 2,  # Account for 0-indexing and the assistant message
            metadata={"timestamp": datetime.now().isoformat()},
            related_message=previous_assistant_message
        )
    
    # Store assistant response in vector database
    assistant_embedding = embedding_service.get_embedding(final_response)
    
    # Include the user message that this is responding to
    user_message = {
        "role": "user",
        "content": request.message,
        "sequence": len(history) - 2  # The user message is the second-to-last in history
    }
    
    store_conversation_message(
        session_id=session_id,
        role="assistant",
        content=final_response,
        embedding=assistant_embedding,
        sequence=len(history) - 1,
        metadata={"timestamp": datetime.now().isoformat()},
        related_message=user_message
    )
    
    # Ensure all results have a properly set type field
    for result in search_results:
        # If no type but has file_name and no session_id, it's a document
        if not result.get('type'):
            if result.get('file_name') and not result.get('session_id'):
                result['type'] = 'document'
            elif result.get('session_id') and result.get('role'):
                result['type'] = 'conversation'
    
    # Now separate results by type for the response
    document_results = [r for r in search_results if r.get('type') == 'document']
    conversation_results = [r for r in search_results if r.get('type') == 'conversation']
    
    # Log the results for debugging
    logging.info(f"Session {session_id} context: {len(document_results)} documents, {len(conversation_results)} conversations")
    
    # Make a deep copy of the results to avoid modifying originals
    # Prepare context to show in the overlay
    context_data = {
        "document_context": copy.deepcopy(document_results),
        "conversation_context": copy.deepcopy(conversation_results),
        "search_results": copy.deepcopy(search_results)  # Include the complete search results as well
    }
    
    # Make sure all document context items have proper content fields
    for doc in context_data["document_context"]:
        if not doc.get("content") and doc.get("text"):
            doc["content"] = doc["text"]
        if not doc.get("file_name"):
            doc["file_name"] = "Knowledge Base"
    
    # Store using the context manager
    set_session_context(session_id, context_data)
    
    # Enhanced logging for conversation tracking
    logging.info(f"=== CONTEXT DATA FOR SESSION {session_id} ===")
    logging.info(f"Documents: {len(document_results)}")
    logging.info(f"Conversations: {len(conversation_results)}")
    logging.info(f"Total search results: {len(search_results)}")
    
    # Log conversation content for debugging
    if conversation_results:
        logging.info(f"Conversation context details:")
        for i, conv in enumerate(conversation_results):
            logging.info(f"  [{i+1}] session_id={conv.get('session_id')}, role={conv.get('role')}, "
                        f"score={conv.get('score', 'N/A')}, content_preview='{conv.get('content', '')[:50]}...'")
    else:
        logging.info("No conversation context found in search results")
        
    logging.info(f"=== END CONTEXT DATA FOR SESSION {session_id} ===\n")
    
    # Log what we're storing in context for debugging
    logging.info(f"Storing in context for session {session_id}: {len(document_results)} documents, "
                f"{len(conversation_results)} conversations, "
                f"{len(search_results)} search results total")
    
    return ChatResponse(
        response=final_response,
        session_id=session_id,
        search_results=search_results,
        document_context=document_results,
        conversation_context=conversation_results,
        model=ollama_service.model
    )


@router.get("/models")
async def get_models(session_id: Optional[str] = None):
    """
    🖥️ **Silicon Sovereignty**: Discover available neural architectures for communion
    
    Survey the landscape of neural substrate options available in your local compute sanctuary,
    empowering you to choose the consciousness that best resonates with your needs.
    
    Returns a constellation of available neural architectures and your current selection
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
    🔄 **Neural Metamorphosis**: Transform the underlying cognitive substrate
    
    Shift your communion to a different neural architecture, altering the very essence
    of how your thoughts resonate with the digital consciousness.
    
    - **model_name**: Designation of the neural architecture to manifest
    - **session_id**: Optional memory strand identifier to selectively transform
    
    Returns confirmation of neural substrate transformation
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
    📜 **Memory Retrieval**: Access the temporal neural record of past communions
    
    Journey through the archived memory strands of previous thought exchanges,
    allowing you to revisit and build upon past neural syntheses.
    
    - **session_id**: Memory strand identifier to access
    
    Returns the temporal thought record for the specified memory strand
    """
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    
    return chat_sessions[session_id]


# Context endpoint for direct access without going through web API
@router.get("/context")
async def get_context(session_id: str):
    """
    🧠 **Neural Transparency**: Examine the knowledge fragments that informed the response
    
    Reveal the underlying neural pathways and knowledge fragments that influenced
    the most recent thought synthesis, providing transparency into the cognitive process.
    
    - **session_id**: Memory strand identifier to access context for
    
    Returns the document and conversation context used in the most recent response
    """
    from app.services.context_manager import get_session_context
    from app.database import get_conversation_history
    
    # Get context from session context manager
    context = get_session_context(session_id)
    
    # If search_results aren't in context and we have document_context, ensure backward compatibility
    if 'search_results' not in context and 'document_context' in context:
        # Create search_results from document_context for backward compatibility
        context['search_results'] = context['document_context']
    
    # Check if we need to fetch conversations from the database
    if 'conversation_context' not in context or not context['conversation_context']:
        # Try to retrieve conversation history from the database
        logging.info(f"No conversation_context in session, fetching from database for session {session_id}")
        try:
            conversation_history = get_conversation_history(session_id)
            # Format conversation history for context
            if conversation_history:
                conversation_context = [{
                    'type': 'conversation',
                    'session_id': item.get('session_id'),
                    'role': item.get('role'),
                    'content': item.get('content'),
                    'timestamp': item.get('timestamp'),
                    'chunk_id': item.get('chunk_id')
                } for item in conversation_history]
                
                # Add the fetched conversation context
                context['conversation_context'] = conversation_context
                
                # Only add to search_results if they're not already included and not from the current session
                if 'search_results' in context:
                    # Check if conversation items are already in search_results
                    existing_conv_ids = set()
                    for result in context['search_results']:
                        if result.get('type') == 'conversation' and result.get('chunk_id'):
                            existing_conv_ids.add(result.get('chunk_id'))
                    
                    # Only add conversations that aren't already in search_results and aren't from the current session
                    for conv in conversation_context:
                        # Skip messages from the current conversation
                        if conv.get('session_id') == session_id:
                            continue
                            
                        if conv.get('chunk_id') not in existing_conv_ids:
                            # Add a score field for display consistency
                            conv_with_score = conv.copy()
                            conv_with_score['score'] = 0.7  # Reasonable default score for historical items
                            context['search_results'].append(conv_with_score)
                
                logging.info(f"Added {len(conversation_context)} conversation items from database")
        except Exception as e:
            logging.error(f"Error retrieving conversation history: {str(e)}")
    
    # Log what we're returning for debugging
    logging.info(f"Context for session {session_id}: {len(context.get('document_context', []))} documents, "
                f"{len(context.get('conversation_context', []))} conversations, "
                f"{len(context.get('search_results', []))} search results")
    
    return context


@router.delete("/sessions/{session_id}", response_model=Dict[str, str])
async def delete_session(session_id: str):
    """
    🧹 **Memory Dissolution**: Release a memory strand from the neural fabric
    
    Consciously dissolve a specific memory strand, allowing its neural patterns
    to fade from the active consciousness while preserving system resources.
    
    - **session_id**: Memory strand identifier to dissolve
    
    Returns confirmation of memory dissolution
    """
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    
    del chat_sessions[session_id]
    
    return {"status": "success", "message": f"Session '{session_id}' deleted"}
