from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import uuid

from app.models.schemas import UploadResponse, ErrorResponse
from app.services.document_processor import DocumentProcessor
from app.config import MAX_FILE_SIZE

router = APIRouter(
    prefix="/documents", 
    tags=["documents"]
)

@router.post("/upload", response_model=UploadResponse, responses={400: {"model": ErrorResponse}})
async def upload_document(file: UploadFile = File(...)):
    """
    🧠 **Neural Ingestion**: Transform static documents into vibrant thought particles
    
    Upload your knowledge artifacts (PDF, TXT) to be embedded into the neural fabric.
    Each document will be processed, chunked, and vectorized for semantic retrieval.
    
    - **file**: Knowledge artifact (PDF or TXT) to be neurally encoded
    
    Returns a unique neural identifier and quantification of thought fragments generated
    """
    processor = DocumentProcessor()
    
    # Check file size
    file_size = 0
    contents = await file.read(MAX_FILE_SIZE + 1)
    file_size = len(contents)
    
    # Reset file pointer
    await file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Maximum file size is {MAX_FILE_SIZE // (1024 * 1024)} MB"
        )
    
    # Check file type
    if not processor.is_allowed_file(file.filename):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed types: PDF, TXT"
        )
    
    try:
        file_id, chunks_count = processor.process_file(file, file.filename)
        
        return UploadResponse(
            file_id=file_id,
            file_name=file.filename,
            chunks=chunks_count,
            message="Document processed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
