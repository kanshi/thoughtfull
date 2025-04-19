import os
import uuid
import PyPDF2
from typing import List, Tuple
import shutil
from app.config import UPLOAD_DIR, CHUNK_SIZE, ALLOWED_EXTENSIONS
from app.services.embedding import EmbeddingService
from app.database import insert_documents
from app.utils.logging import get_logger

class DocumentProcessor:
    def __init__(self):
        """Initialize document processor"""
        self.embedding_service = EmbeddingService()
        self.logger = get_logger(__name__)
        
        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        self.logger.info(f"Document processor initialized with upload directory: {UPLOAD_DIR}")
        
    def is_allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
        
    def process_file(self, file, filename: str) -> Tuple[str, int]:
        """
        Process uploaded file: save, extract text, chunk, embed, and store in database
        
        Args:
            file: File-like object
            filename: Original filename
            
        Returns:
            Tuple[str, int]: (file_id, number of chunks processed)
        """
        self.logger.info(f"Processing file: {filename}")
        
        if not self.is_allowed_file(filename):
            self.logger.warning(f"Rejected file with unsupported type: {filename}")
            raise ValueError(f"Unsupported file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        self.logger.info(f"Generated file ID: {file_id} for {filename}")
        
        # Save file to disk
        file_extension = filename.rsplit('.', 1)[1].lower()
        saved_path = os.path.join(UPLOAD_DIR, f"{file_id}.{file_extension}")
        
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        self.logger.info(f"Saved file to: {saved_path}")
        
        # Extract text based on file type
        self.logger.info(f"Extracting text from {file_extension} file")
        if file_extension == "pdf":
            text = self._extract_text_from_pdf(saved_path)
        elif file_extension == "txt":
            text = self._extract_text_from_txt(saved_path)
        else:
            # This should never happen due to the is_allowed_file check
            self.logger.error(f"Unsupported file type encountered: {file_extension}")
            raise ValueError(f"Unsupported file type: {file_extension}")
            
        # Chunk text into smaller pieces
        self.logger.info(f"Chunking extracted text from {filename}")
        chunks = self._chunk_text(text)
        
        if not chunks:
            self.logger.warning(f"No text content could be extracted from file: {filename}")
            raise ValueError("No text content could be extracted from the file")
        
        self.logger.info(f"Generated {len(chunks)} chunks from {filename}")
        
        # Get embeddings for chunks
        self.logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = self.embedding_service.get_embeddings(chunks)
        
        # Store in database
        self.logger.info(f"Storing {len(chunks)} chunks in database for {filename}")
        num_chunks = insert_documents(chunks, embeddings, file_id, filename)
        
        self.logger.info(f"Successfully processed {filename}: stored {num_chunks} chunks with ID {file_id}")
        return file_id, num_chunks
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                self.logger.info(f"Extracting text from PDF with {total_pages} pages: {file_path}")
                
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    
                    # Log progress for large PDFs
                    if total_pages > 10 and page_num % 10 == 0 and page_num > 0:
                        self.logger.info(f"Extracted {page_num}/{total_pages} pages from {file_path}")
                        
                self.logger.info(f"Completed text extraction from PDF: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to extract text from PDF {file_path}: {str(e)}", exc_info=True)
            raise ValueError(f"Error extracting text from PDF: {str(e)}")
        return text
    
    def _extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        self.logger.info(f"Extracting text from TXT file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                self.logger.info(f"Successfully extracted text with UTF-8 encoding from: {file_path}")
                return content
        except UnicodeDecodeError:
            # Try with a different encoding if utf-8 fails
            self.logger.warning(f"UTF-8 decoding failed for {file_path}, trying latin-1 encoding")
            with open(file_path, "r", encoding="latin-1") as file:
                content = file.read()
                self.logger.info(f"Successfully extracted text with latin-1 encoding from: {file_path}")
                return content
        except Exception as e:
            self.logger.error(f"Failed to extract text from TXT {file_path}: {str(e)}", exc_info=True)
            raise ValueError(f"Error extracting text from TXT: {str(e)}")
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of specified size"""
        if not text or text.isspace():
            self.logger.warning("Received empty or whitespace-only text for chunking")
            return []
        
        self.logger.info(f"Chunking text of length {len(text)} characters with chunk size {CHUNK_SIZE}")
        
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        self.logger.info(f"Text split into {len(paragraphs)} paragraphs")
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If adding this paragraph exceeds chunk size, store current chunk and start new one
            if len(current_chunk) + len(paragraph) > CHUNK_SIZE and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + " "
            else:
                current_chunk += paragraph + " "
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        self.logger.info(f"Initial chunking created {len(chunks)} chunks")
        
        # Handle case where chunks are still too large
        result_chunks = []
        oversized_chunks = 0
        
        for chunk in chunks:
            if len(chunk) > CHUNK_SIZE:
                # Further split into smaller chunks
                oversized_chunks += 1
                sub_chunks = []
                for i in range(0, len(chunk), CHUNK_SIZE):
                    sub_chunks.append(chunk[i:i+CHUNK_SIZE].strip())
                result_chunks.extend(sub_chunks)
                self.logger.info(f"Split oversized chunk of {len(chunk)} chars into {len(sub_chunks)} sub-chunks")
            else:
                result_chunks.append(chunk)
        
        if oversized_chunks > 0:
            self.logger.info(f"Found {oversized_chunks} oversized chunks that needed further splitting")
        
        self.logger.info(f"Final chunking result: {len(result_chunks)} chunks")
        return result_chunks
