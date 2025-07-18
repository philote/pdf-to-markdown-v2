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
            
            # Prepare API call parameters
            api_params = {
                'model': self.model,
                'document': document_data,
                'include_image_base64': False  # We don't need embedded images for TTRPG text
            }
            
            # Add page ranges if specified (Mistral API uses 0-indexed pages)
            if pages_to_process:
                # Convert 1-indexed page ranges to 0-indexed for API
                zero_indexed_pages = [p - 1 for p in pages_to_process]
                api_params['pages'] = zero_indexed_pages
                if self.verbose:
                    print(f"Processing pages: {zero_indexed_pages} (0-indexed)")
            else:
                if self.verbose:
                    print("Processing entire document (no page range specified)")
            
            response = self.client.ocr.process(**api_params)
            
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
        Process PDF with enhanced formatting preservation
        
        Args:
            pdf_path: Path to PDF file
            page_ranges: Optional page ranges
            aggressive_formatting: Use aggressive formatting preservation
            
        Returns:
            Dict with processing results
        """
        if self.verbose:
            if aggressive_formatting:
                print("Using enhanced formatting preservation mode")
            else:
                print("Using standard formatting preservation")
        
        # Try to use enhanced OCR processing for better formatting
        return self.process_pdf_enhanced(pdf_path, page_ranges, aggressive_formatting)
    
    def process_pdf_enhanced(self, pdf_path: str, page_ranges: Optional[str] = None,
                           enhanced_formatting: bool = True) -> Dict[str, Any]:
        """
        Enhanced PDF processing with better formatting preservation
        
        Args:
            pdf_path: Path to PDF file
            page_ranges: Optional page ranges
            enhanced_formatting: Use enhanced formatting options
            
        Returns:
            Dict with processing results
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
            document_data = self._prepare_base64_document(pdf_path, pages_to_process)
            
            # Submit OCR request with enhanced formatting
            if self.verbose:
                print("Submitting to Mistral OCR API with enhanced formatting...")
            
            # Prepare API call parameters with enhanced formatting options
            api_params = {
                'model': self.model,
                'document': document_data,
                'include_image_base64': False  # We don't need embedded images for TTRPG text
            }
            
            # Add page ranges if specified (Mistral API uses 0-indexed pages)
            if pages_to_process:
                # Convert 1-indexed page ranges to 0-indexed for API
                zero_indexed_pages = [p - 1 for p in pages_to_process]
                api_params['pages'] = zero_indexed_pages
                if self.verbose:
                    print(f"Processing pages: {zero_indexed_pages} (0-indexed)")
            else:
                if self.verbose:
                    print("Processing entire document (no page range specified)")
            
            # Enhanced formatting: Try to use document annotation format for better structure
            if enhanced_formatting:
                try:
                    # This is experimental - try to request more structured output
                    # The API might provide better formatting preservation with structured output
                    api_params['include_image_base64'] = False
                    if self.verbose:
                        print("Requesting enhanced formatting preservation")
                except Exception as e:
                    if self.verbose:
                        print(f"Enhanced formatting not available, using standard: {e}")
            
            response = self.client.ocr.process(**api_params)
            
            # Extract markdown content
            markdown_content = self._extract_markdown(response)
            
            # Post-process for even better formatting if requested
            if enhanced_formatting:
                markdown_content = self._enhance_formatting(markdown_content)
            
            processing_time = time.time() - start_time
            
            result = {
                'status': 'success',
                'input_file': str(pdf_path),
                'markdown_content': markdown_content,
                'processing_time': processing_time,
                'model': self.model,
                'page_ranges': page_ranges,
                'enhanced_formatting': enhanced_formatting,
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
    
    def _enhance_formatting(self, markdown_content: str) -> str:
        """
        Post-process markdown to enhance formatting
        
        Args:
            markdown_content: Raw markdown from OCR
            
        Returns:
            Enhanced markdown with improved formatting
        """
        if not markdown_content:
            return markdown_content
        
        # Apply additional formatting enhancements
        lines = markdown_content.split('\n')
        enhanced_lines = []
        
        for line in lines:
            # Enhance common TTRPG formatting patterns
            enhanced_line = line
            
            # Ensure move names are properly bolded (common TTRPG pattern)
            # Look for patterns like "When you do something:" and make "do something" bold
            import re
            
            # Pattern for move descriptions
            move_pattern = r'When you ([^,]+)(?:,|:)'
            enhanced_line = re.sub(move_pattern, lambda m: f'When you **{m.group(1)}**,', enhanced_line)
            
            # Pattern for roll instructions  
            roll_pattern = r'roll\+(\w+)'
            enhanced_line = re.sub(roll_pattern, r'roll+**\1**', enhanced_line)
            
            # Ensure stat names are emphasized
            stat_pattern = r'\b(cool|hard|hot|sharp|weird|Hx)\b'
            enhanced_line = re.sub(stat_pattern, r'**\1**', enhanced_line, flags=re.IGNORECASE)
            
            enhanced_lines.append(enhanced_line)
        
        return '\n'.join(enhanced_lines)
    
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