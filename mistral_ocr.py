#!/usr/bin/env python3
"""
Mistral OCR Integration Module

Handles PDF to Markdown conversion using Mistral's OCR API
"""

import os
import base64
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from mistralai import Mistral
from mistralai.models.ocrrequest import DocumentURLChunk
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MistralOCR:
    """Mistral OCR client for PDF to Markdown conversion"""
    
    def __init__(self, api_key: Optional[str] = None, verbose: bool = True):
        """
        Initialize Mistral OCR client
        
        Args:
            api_key: Mistral API key (defaults to MISTRAL_API_KEY env var)
            verbose: Enable verbose logging
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        
        self.client = Mistral(api_key=self.api_key)
        self.verbose = verbose
        self.model = "mistral-ocr-latest"
        
    def process_pdf(self, pdf_path: str, page_ranges: Optional[str] = None,
                   use_base64: bool = True) -> Dict[str, Any]:
        """
        Process PDF through Mistral OCR
        
        Args:
            pdf_path: Path to PDF file
            page_ranges: Optional page ranges (e.g., "1-10,15-20")
            use_base64: Use base64 encoding (True) or URL upload (False)
            
        Returns:
            Dict with processing results including markdown content
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        start_time = time.time()
        
        if self.verbose:
            print(f"Processing PDF: {pdf_path.name}")
            if page_ranges:
                print(f"Page ranges: {page_ranges}")
        
        try:
            # Parse page ranges if provided
            pages_to_process = self._parse_page_ranges(page_ranges) if page_ranges else None
            
            # Prepare document for OCR
            if use_base64:
                document_data = self._prepare_base64_document(pdf_path, pages_to_process)
            else:
                # For URL upload, would need to implement file upload service
                raise NotImplementedError("URL upload not yet implemented")
            
            # Submit OCR request
            if self.verbose:
                print("Submitting to Mistral OCR API...")
            
            response = self.client.ocr.process(
                model=self.model,
                document=document_data,
                include_image_base64=False  # We don't need embedded images for TTRPG text
            )
            
            # Extract markdown content
            markdown_content = self._extract_markdown(response)
            
            processing_time = time.time() - start_time
            
            result = {
                'status': 'success',
                'input_file': str(pdf_path),
                'markdown_content': markdown_content,
                'processing_time': processing_time,
                'model': self.model,
                'page_ranges': page_ranges,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.verbose:
                print(f"OCR completed in {processing_time:.1f} seconds")
            
            return result
            
        except Exception as e:
            error_result = {
                'status': 'error',
                'input_file': str(pdf_path),
                'error': str(e),
                'error_type': type(e).__name__,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.verbose:
                print(f"OCR failed: {e}")
            
            return error_result
    
    def process_with_formatting_prompt(self, pdf_path: str, page_ranges: Optional[str] = None,
                                     aggressive_formatting: bool = True) -> Dict[str, Any]:
        """
        Process PDF with specific formatting preservation prompt
        
        Args:
            pdf_path: Path to PDF file
            page_ranges: Optional page ranges
            aggressive_formatting: Use aggressive formatting preservation
            
        Returns:
            Dict with processing results
        """
        # Prepare the formatting prompt based on requirements
        if aggressive_formatting:
            # Adapted prompt for Mistral OCR based on the README suggestion
            prompt = """Extract text with MANDATORY formatting preservation:
- Convert ALL bold text to **bold** markdown
- Convert ALL italic text to *italic* markdown  
- Preserve headers with appropriate # levels
- Maintain list structures and indentation
- Keep table formatting intact
- NEVER skip formatting - check every word for font weight and style changes
- Bold headings, game titles, author names, and emphasized terms are CRITICAL"""
        else:
            prompt = "Extract text preserving basic formatting and structure"
        
        # For now, Mistral OCR doesn't support custom prompts in the basic API
        # This would need to be implemented if/when the API supports it
        # For now, we'll use the standard processing
        
        if self.verbose and aggressive_formatting:
            print("Note: Using standard Mistral OCR formatting (custom prompts not yet supported)")
        
        return self.process_pdf(pdf_path, page_ranges)
    
    def _prepare_base64_document(self, pdf_path: Path, pages: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Prepare document data with base64 encoding
        
        Args:
            pdf_path: Path to PDF file
            pages: Optional list of page numbers to include
            
        Returns:
            Document data dict for API
        """
        # If specific pages requested, we need to extract them first
        if pages:
            # This would require using PyPDF2 or fitz to extract specific pages
            # For now, we'll process the entire document
            if self.verbose:
                print("Note: Page range extraction not yet implemented, processing entire document")
        
        # Read and encode PDF
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()
        
        base64_data = base64.b64encode(pdf_data).decode('utf-8')
        
        return DocumentURLChunk(
            document_url=f"data:application/pdf;base64,{base64_data}",
            document_name=pdf_path.name
        )
    
    def _parse_page_ranges(self, page_ranges: str) -> List[int]:
        """
        Parse page range string into list of page numbers
        
        Args:
            page_ranges: String like "1-5,10,15-20"
            
        Returns:
            List of page numbers
        """
        pages = []
        
        for part in page_ranges.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                pages.extend(range(start, end + 1))
            else:
                pages.append(int(part))
        
        return sorted(set(pages))  # Remove duplicates and sort
    
    def _extract_markdown(self, response) -> str:
        """
        Extract markdown content from OCR response
        
        Args:
            response: Mistral OCR API response
            
        Returns:
            Markdown string
        """
        # The response structure includes pages with markdown content
        try:
            # Check if response has pages attribute
            if hasattr(response, 'pages'):
                # Extract markdown from all pages and combine
                markdown_content = []
                for page in response.pages:
                    if hasattr(page, 'markdown') and page.markdown:
                        markdown_content.append(page.markdown)
                return '\n\n---\n\n'.join(markdown_content)
            
            # Legacy handling for different response structures
            elif hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'content'):
                return response.content
            elif isinstance(response, dict):
                # Handle dict response
                if 'pages' in response:
                    markdown_content = []
                    for page in response['pages']:
                        if 'markdown' in page and page['markdown']:
                            markdown_content.append(page['markdown'])
                    return '\n\n---\n\n'.join(markdown_content)
                return response.get('text', '') or response.get('content', '')
            else:
                # Fallback - convert response to string
                return str(response)
                
        except Exception as e:
            if self.verbose:
                print(f"Warning: Error extracting markdown: {e}")
            return str(response)
    
    def check_capabilities(self) -> Dict[str, Any]:
        """
        Check OCR model capabilities and status
        
        Returns:
            Dict with capability information
        """
        try:
            # This would check model availability, limits, etc.
            # For now, return basic info
            return {
                'model': self.model,
                'status': 'available',
                'features': [
                    'pdf_processing',
                    'markdown_output',
                    'formatting_preservation',
                    'multi_language',
                    'table_extraction'
                ]
            }
        except Exception as e:
            return {
                'model': self.model,
                'status': 'error',
                'error': str(e)
            }


def main():
    """Test the OCR module with a simple example"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mistral_ocr.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    try:
        # Initialize OCR client
        ocr = MistralOCR(verbose=True)
        
        # Check capabilities
        print("Checking OCR capabilities...")
        caps = ocr.check_capabilities()
        print(f"Model: {caps['model']}, Status: {caps['status']}")
        
        # Process PDF
        print("\n" + "="*50)
        result = ocr.process_pdf(pdf_path)
        
        if result['status'] == 'success':
            print(f"\nOCR completed successfully!")
            print(f"Processing time: {result['processing_time']:.1f}s")
            print(f"\nMarkdown preview (first 500 chars):")
            print("-" * 50)
            print(result['markdown_content'][:500] + "...")
        else:
            print(f"\nOCR failed: {result['error']}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()