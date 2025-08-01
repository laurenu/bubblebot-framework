"""
Bubblebot Framework - FastAPI Application

Main application entry point for the Bubblebot chatbot framework.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from pathlib import Path
import tempfile
from typing import List, Dict, Any
from datetime import datetime

from app.services.document_processor import DocumentProcessor, ProcessingResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🫧 Bubblebot Framework",
    description="Multi-tenant AI chatbot framework for real estate professionals",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_processor = DocumentProcessor()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("🫧 Bubblebot Framework starting up...")
    logger.info("✅ Document processor initialized")

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("👋 Bubblebot Framework shutting down...")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "🫧 Welcome to Bubblebot Framework!",
        "version": "0.1.0",
        "docs": "/docs",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "services": {
            "document_processor": "operational",
            "database": "not_connected",  # Will be implemented in later days
            "ai_service": "not_connected"  # Will be implemented in later days
        }
    }

@app.post("/api/v1/documents/process")
async def process_document(
    file: UploadFile = File(...),
    tenant_id: str = "default_tenant"  # In real app, get from authentication
):
    """
    Process an uploaded document and extract text chunks.
    
    Args:
        file: The uploaded document file
        tenant_id: ID of the tenant uploading the document
        
    Returns:
        ProcessingResult with chunks and metadata
    """
    # Validate file type
    allowed_extensions = {'.txt', '.pdf', '.docx'}
    file_suffix = Path(file.filename).suffix.lower()
    
    if file_suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_suffix}. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
        # Write uploaded content to temp file
        content = await file.read()
        temp_file.write(content)
        temp_path = Path(temp_file.name)
    
    try:
        # Process the document
        result = await document_processor.process_file(temp_path, tenant_id)
        
        if result.success:
            logger.info(f"Successfully processed {file.filename}: {result.total_chunks} chunks")
            
            # Convert result to JSON-serializable format
            return {
                "success": True,
                "filename": file.filename,
                "tenant_id": tenant_id,
                "total_chunks": result.total_chunks,
                "total_words": result.total_words,
                "processing_time_seconds": result.processing_time_seconds,
                "chunks": [
                    {
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "word_count": chunk.word_count,
                        "source_file": chunk.source_file,
                        "document_type": chunk.document_type.value,
                        "metadata": chunk.metadata
                    }
                    for chunk in result.chunks
                ]
            }
        else:
            logger.error(f"Failed to process {file.filename}: {result.error_message}")
            raise HTTPException(
                status_code=422,
                detail=f"Document processing failed: {result.error_message}"
            )
    
    except Exception as e:
        logger.error(f"Unexpected error processing {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    
    finally:
        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()

@app.get("/api/v1/documents/stats")
async def get_processing_stats():
    """
    Get document processing statistics.
    
    Note: In a real application, this would query the database
    for historical processing data. For now, returns sample data.
    """
    # This is demo data - in real app, query from database
    sample_results = [
        ProcessingResult(
            success=True,
            chunks=[],
            total_chunks=5,
            total_words=1250,
            processing_time_seconds=1.2
        ),
        ProcessingResult(
            success=True,
            chunks=[],
            total_chunks=3,
            total_words=780,
            processing_time_seconds=0.8
        ),
        ProcessingResult(
            success=False,
            chunks=[],
            total_chunks=0,
            total_words=0,
            error_message="Sample error",
            processing_time_seconds=0.1
        )
    ]
    
    stats = document_processor.get_processing_stats(sample_results)
    return {
        "message": "Document processing statistics",
        "stats": stats,
        "note": "This is sample data. In production, this would query real processing history."
    }

@app.get("/api/v1/tenants/{tenant_id}/documents")
async def get_tenant_documents(tenant_id: str):
    """
    Get documents for a specific tenant.
    
    Args:
        tenant_id: The tenant identifier
        
    Returns:
        List of documents for the tenant
        
    Note: This is a placeholder endpoint. In production, this would
    query the database for actual tenant documents.
    """
    return {
        "tenant_id": tenant_id,
        "documents": [],
        "message": "Document storage will be implemented in Day 2-3 with database integration."
    }

@app.post("/api/v1/tenants/{tenant_id}/documents/batch")
async def process_batch_documents(
    tenant_id: str,
    files: List[UploadFile] = File(...)
):
    """
    Process multiple documents in batch for a tenant.
    
    Args:
        tenant_id: The tenant identifier
        files: List of uploaded document files
        
    Returns:
        Batch processing results
    """
    if len(files) > 10:  # Reasonable batch limit
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 10 files per batch."
        )
    
    results = []
    temp_files = []
    
    try:
        for file in files:
            # Validate file type
            allowed_extensions = {'.txt', '.pdf', '.docx'}
            file_suffix = Path(file.filename).suffix.lower()
            
            if file_suffix not in allowed_extensions:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": f"Unsupported file type: {file_suffix}"
                })
                continue
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_path = Path(temp_file.name)
                temp_files.append(temp_path)
            
            # Process the document
            result = await document_processor.process_file(temp_path, tenant_id)
            
            if result.success:
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "total_chunks": result.total_chunks,
                    "total_words": result.total_words,
                    "processing_time_seconds": result.processing_time_seconds
                })
            else:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": result.error_message
                })
    
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        )
    
    finally:
        # Cleanup temp files
        for temp_path in temp_files:
            if temp_path.exists():
                temp_path.unlink()
    
    # Generate batch statistics
    successful_files = [r for r in results if r.get("success", False)]
    failed_files = [r for r in results if not r.get("success", False)]
    
    return {
        "tenant_id": tenant_id,
        "batch_summary": {
            "total_files": len(files),
            "successful": len(successful_files),
            "failed": len(failed_files),
            "total_chunks": sum(r.get("total_chunks", 0) for r in successful_files),
            "total_words": sum(r.get("total_words", 0) for r in successful_files)
        },
        "file_results": results
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "message": "The requested resource was not found.",
            "path": str(request.url.path)
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler."""
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )

if __name__ == "__main__":
    # For development - in production, use gunicorn or similar
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
