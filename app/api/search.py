from fastapi import APIRouter, Query, HTTPException

from app.models.schemas import SearchResponse, SearchResult
from app.services.embedding import EmbeddingService
from app.database import search_documents

router = APIRouter(
    prefix="/search", 
    tags=["search"]
)

@router.get("/", response_model=SearchResponse)
async def search(
    query: str = Query(..., description="Search query"),
    limit: int = Query(5, ge=1, le=100, description="Maximum number of results to return")
):
    """
    🔮 **Neural Resonance**: Traverse the semantic landscape of your knowledge corpus
    
    Your query becomes a thought-vector, journeying through the neural pathways
    of your document dataverse to find resonant thought fragments.
    
    - **query**: Thought pattern to find semantic resonances for
    - **limit**: Maximum number of neural echoes to return (default: 5)
    
    Returns thought fragments that vibrate at frequencies similar to your query
    """
    if not query or query.isspace():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")
    
    # Generate embedding for query
    embedding_service = EmbeddingService()
    query_embedding = embedding_service.get_embedding(query)
    
    # Search in vector database
    results = search_documents(query_embedding, limit)
    
    if not results or len(results) == 0:
        return SearchResponse(
            results=[],
            count=0,
            query=query
        )
    
    # Format results
    formatted_results = []
    for hits in results:
        for hit in hits:
            formatted_results.append(SearchResult(
                file_id=hit.entity.get('file_id'),
                file_name=hit.entity.get('file_name'),
                content=hit.entity.get('content'),
                chunk_id=hit.entity.get('chunk_id'),
                score=float(hit.score)
            ))
    
    return SearchResponse(
        results=formatted_results,
        count=len(formatted_results),
        query=query
    )
