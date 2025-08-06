"""
Integration tests for the document processing pipeline.
"""

import asyncio
import tempfile
from pathlib import Path
import pytest

from app.services.document_processor import DocumentProcessor, DocumentType

class TestDocumentProcessorIntegration:
    """Integration tests for complete document processing workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_processing_workflow(self):
        """Test the complete document processing workflow."""
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        
        # Create a realistic real estate FAQ document
        real_estate_content = """
        Frequently Asked Questions - Home Buying Process
        
        What is the first step in buying a home?
        The first step is getting pre-approved for a mortgage. This involves submitting financial documents to a lender who will determine how much you can borrow. Pre-approval gives you a clear budget and shows sellers you're a serious buyer.
        
        How much should I save for a down payment?
        Most conventional loans require 10-20% down, but FHA loans allow as little as 3.5%. Don't forget to budget for closing costs, which typically run 2-5% of the home's purchase price.
        
        What is a home inspection?
        A home inspection is a thorough examination of the property's condition, typically conducted by a licensed inspector. It covers structural elements, electrical systems, plumbing, HVAC, and more. This usually happens during the contingency period after your offer is accepted.
        
        How long does the closing process take?
        From contract to closing typically takes 30-45 days, though it can vary based on financing, inspections, and other factors. Cash purchases can close much faster, sometimes in as little as 1-2 weeks.
        """.strip()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(real_estate_content)
            test_file = Path(f.name)
        
        try:
            # Process the document
            result = await processor.process_file(test_file, "integration_test_tenant")
            
            # Verify successful processing
            assert result.success
            assert result.total_chunks > 1  # Should be chunked due to length
            assert result.total_words > 50
            assert result.processing_time_seconds > 0
            
            # Verify chunks contain expected content
            all_content = " ".join(chunk.content for chunk in result.chunks)
            assert "pre-approved for a mortgage" in all_content
            assert "home inspection" in all_content.lower()
            assert "closing process" in all_content.lower()
            
            # Verify chunk metadata
            for chunk in result.chunks:
                assert chunk.source_file == test_file.name
                assert chunk.document_type == DocumentType.TXT
                assert chunk.metadata["tenant_id"] == "integration_test_tenant"
                assert chunk.word_count > 0
                assert isinstance(chunk.chunk_index, int)
            
            # Verify chunk ordering
            chunk_indices = [chunk.chunk_index for chunk in result.chunks]
            assert chunk_indices == list(range(len(chunk_indices)))
            
        finally:
            test_file.unlink()
    
    @pytest.mark.asyncio
    async def test_multiple_document_processing(self):
        """Test processing multiple documents and generating stats."""
        processor = DocumentProcessor()
        
        # Create multiple test documents
        documents = [
            ("doc1.txt", "This is the first document with some real estate information."),
            ("doc2.txt", "The second document contains different property details and market analysis."),
            ("doc3.txt", "")  # Empty document to test failure handling
        ]
        
        temp_files = []
        results = []
        
        try:
            # Create temporary files and process them
            for filename, content in documents:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(content)
                    f.flush()
                    temp_file = Path(f.name)
                    temp_files.append(temp_file)
                
                result = await processor.process_file(temp_file, f"tenant_{filename}")
                results.append(result)
            
            # Generate and verify stats
            stats = processor.get_processing_stats(results)
            
            assert stats["total_documents"] == 3
            assert stats["successful_documents"] == 2  # First two should succeed
            assert stats["failed_documents"] == 1      # Empty file should fail
            assert stats["success_rate"] == pytest.approx(66.67, rel=1e-2)
            assert stats["total_chunks"] >= 2  # At least one chunk per successful doc
            assert stats["total_words"] > 0
            
        finally:
            # Clean up temp files
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
