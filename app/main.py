from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import pathlib
import logging

from app.api import documents, search, web, chat, web_stream, web_refresh
from app.database import connect_to_milvus, create_collection_if_not_exists
from app.config import UPLOAD_DIR
from app.utils.logging import setup_logging, get_logger

# Setup logging first
setup_logging(log_level="INFO", enable_json_logs=False, log_to_file=True)
logger = get_logger(__name__)

# Setup templates and static files
BASE_DIR = pathlib.Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(
    title="ThoughtFull - Semantic Search API",
    description="A semantic search service for private documents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(search.router)
app.include_router(web.router)
app.include_router(chat.router)
app.include_router(web_stream.router)
app.include_router(web_refresh.router)

# Mount static files directory
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {str(exc)}", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"error": f"An unexpected error occurred: {str(exc)}"}
    )

@app.on_event("startup")
async def startup_event():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Log configuration values at startup
    from app.config import MILVUS_HOST, MILVUS_PORT
    logger.info(f"Starting application with Milvus configuration: {MILVUS_HOST}:{MILVUS_PORT}")
    
    try:
        connect_to_milvus()
        create_collection_if_not_exists()
        logger.info("Successfully connected to Milvus and initialized collection")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {str(e)}", exc_info=e)
        logger.warning("Please ensure Milvus is running before starting the application")

@app.get("/", response_class=HTMLResponse, tags=["web"])
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "ThoughtFull - Semantic Search"}
    )

@app.get("/health", tags=["status"])
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
