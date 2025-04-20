import uuid
import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any

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
    post_pipeline.add_step(VectorSearchStep(max_results=MAX_CONTEXT_CHUNKS))
    
    # Process the LLM response through the post-pipeline
    post_context = {
        'message': assistant_message,
        'message_embedding': response_embedding
    }
    post_results = post_pipeline.process(post_context)
    
    # Extract post-search results
    post_search_results = post_results.get('search_results', [])
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
    
    return ChatResponse(
        response=final_response,
        session_id=session_id,
        search_results=search_results,
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
