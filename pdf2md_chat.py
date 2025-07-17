#!/usr/bin/env python3
"""
Test script for Mistral Chat API with document processing for formatting preservation.

This approach uses Mistral's chat interface combined with document processing
to leverage custom prompts for better formatting preservation.
"""

import os
import sys
import json
import time
import tempfile
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import fitz  # PyMuPDF
from mistralai import Mistral
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

from pdf_optimizer import PDFOptimizer

# Load environment variables
load_dotenv()

console = Console()


class DummyProgress:
    """Dummy progress class that provides the same interface as Rich Progress but does nothing."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def add_task(self, description, total=None):
        return "dummy_task"
    
    def update(self, task_id, description=None, advance=None, **kwargs):
        pass


class APITimeoutError(Exception):
    """Raised when API call exceeds timeout"""
    pass


class TimeoutContext:
    """Context manager for API call timeouts"""
    def __init__(self, timeout_seconds: int = 180):  # 3 minute default
        self.timeout_seconds = timeout_seconds
        self.old_handler = None
    
    def __enter__(self):
        def timeout_handler(signum, frame):
            raise APITimeoutError(f"API call timed out after {self.timeout_seconds} seconds")
        
        self.old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout_seconds)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)
        if self.old_handler:
            signal.signal(signal.SIGALRM, self.old_handler)


class MistralChatOCR:
    """
    Mistral Chat API integration for document processing with custom formatting prompts.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "mistral-small-latest", verbose: bool = True, enable_fallback: bool = True):
        """
        Initialize the Mistral Chat OCR client.
        
        Args:
            api_key: Mistral API key (if None, loads from environment)
            model: Model to use for chat completions
            verbose: Enable verbose output
            enable_fallback: Enable automatic fallback to OCR API on Chat API failures
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        
        self.client = Mistral(api_key=self.api_key)
        self.model = model
        self.verbose = verbose
        self.enable_fallback = enable_fallback
        self.jobs_dir = Path(".jobs")
        self.jobs_dir.mkdir(exist_ok=True)
        
        # Initialize PDF optimizer
        self.optimizer = PDFOptimizer(verbose=False)  # We'll handle output ourselves
        
        # Initialize OCR fallback if enabled
        self.ocr_fallback = None
        if enable_fallback:
            try:
                from mistral_ocr import MistralOCR
                self.ocr_fallback = MistralOCR(api_key=self.api_key, verbose=False)
                if self.verbose:
                    console.print(f"[green]Initialized Mistral Chat OCR with OCR fallback enabled[/green]")
            except ImportError:
                if self.verbose:
                    console.print(f"[yellow]OCR fallback not available (mistral_ocr.py not found)[/yellow]")
        
        if self.verbose and not enable_fallback:
            console.print(f"[green]Initialized Mistral Chat OCR with model: {self.model}[/green]")
    
    def process_pdf_with_chat(self, pdf_path: str, formatting_prompt: Optional[str] = None, 
                            page_ranges: Optional[str] = None, skip_optimization: bool = False, 
                            keep_images: bool = False, chunk_size: int = 15, 
                            chunk_threshold: int = 20, disable_chunking: bool = False,
                            show_progress: bool = True) -> Dict[str, Any]:
        """
        Process PDF using Mistral Chat API with custom formatting instructions.
        
        Args:
            pdf_path: Path to PDF file
            formatting_prompt: Custom prompt for formatting preservation
            page_ranges: Optional page ranges (e.g., "1-3,5,7-9")
            skip_optimization: Skip PDF optimization step
            keep_images: Keep images during optimization
            chunk_size: Pages per chunk for large documents (default: 15)
            chunk_threshold: Page count threshold to trigger chunking (default: 20)
            disable_chunking: Disable automatic chunking for large documents
            
        Returns:
            Dict with processing results
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Initialize job tracking
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{pdf_path.stem}"
        job_data = {
            'id': job_id,
            'input_file': str(pdf_path),
            'status': 'started',
            'start_time': datetime.now().isoformat(),
            'page_ranges': page_ranges,
            'formatting_prompt_type': 'custom' if formatting_prompt else 'default'
        }
        self._save_job(job_id, job_data)
        
        # Extract specific pages if page_ranges provided
        original_pdf_path = pdf_path
        if page_ranges:
            pdf_path = self._extract_pages(pdf_path, page_ranges)
            temp_file = True
        else:
            temp_file = False
        
        # Default formatting prompt if none provided
        if not formatting_prompt:
            formatting_prompt = self._get_default_formatting_prompt()
        
        start_time = time.time()
        
        if self.verbose:
            console.print(f"[blue]Processing PDF: {pdf_path.name}[/blue]")
            console.print(f"[blue]Using formatting prompt: {len(formatting_prompt)} characters[/blue]")
        
        # Create progress context only if requested
        if show_progress:
            progress_context = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console
            )
        else:
            progress_context = None
        
        with progress_context if show_progress else DummyProgress() as progress:
            
            # Overall task with 5 main steps (analysis, optimization, upload, process, finalize)
            main_task = progress.add_task("[cyan]Converting PDF to Markdown", total=5)
            
            try:
                # Step 1: Analyze PDF
                progress.update(main_task, description="[cyan]Analyzing PDF...", advance=1)
                analysis = self.optimizer.analyze_pdf(str(pdf_path))
                job_data['analysis'] = analysis
                
                # Step 2: Optimize if needed
                optimized_path = pdf_path
                optimization_result = None
                
                if not skip_optimization and self.optimizer.should_optimize(str(pdf_path)):
                    progress.update(main_task, description="[yellow]Optimizing PDF...", advance=1)
                    optimization_result = self.optimizer.optimize_for_text_extraction(
                        str(pdf_path),
                        remove_images=not keep_images
                    )
                    
                    if optimization_result['status'] == 'success':
                        optimized_path = Path(optimization_result['output_file'])
                        job_data['optimization'] = optimization_result
                        if self.verbose:
                            console.print(f"[green]PDF optimized: {optimization_result['size_reduction_percent']:.1f}% size reduction[/green]")
                else:
                    progress.update(main_task, description="[green]Skipping optimization", advance=1)
                    job_data['optimization'] = {'skipped': True}
                
                # Step 3: Check if chunking is needed
                total_pages = self._get_pdf_page_count(optimized_path)
                needs_chunking = (
                    not disable_chunking and 
                    not page_ranges and  # Don't chunk if user specified page ranges
                    total_pages > chunk_threshold
                )
                
                if needs_chunking:
                    if self.verbose:
                        console.print(f"[cyan]Document has {total_pages} pages, using chunking strategy (chunk size: {chunk_size})[/cyan]")
                    progress.update(main_task, description="[blue]Processing in chunks...", advance=1)
                    
                    # Process document in chunks, passing progress context
                    result = self._process_pdf_chunked(
                        optimized_path, formatting_prompt, job_id, job_data, progress, main_task
                    )
                    return result
                
                else:
                    if self.verbose and not page_ranges:
                        console.print(f"[green]Document has {total_pages} pages, processing as single document[/green]")
                    progress.update(main_task, description="[blue]Processing single document...", advance=1)
                
                # Step 4: Upload PDF file (for single document processing)
                progress.update(main_task, description="[yellow]Uploading PDF file...", advance=1)
                
                with open(optimized_path, "rb") as f:
                    uploaded_pdf = self.client.files.upload(
                        file={
                            "file_name": optimized_path.name,
                            "content": f,
                        },
                        purpose="ocr"
                    )
                
                # Get signed URL for the uploaded file
                signed_url = self.client.files.get_signed_url(file_id=uploaded_pdf.id)
                
                # Update job with upload details
                job_data['mistral_api'] = {
                    'file_upload_id': uploaded_pdf.id,
                    'signed_url': signed_url.url,
                    'upload_timestamp': datetime.now().isoformat()
                }
                self._save_job(job_id, job_data)
                
                if self.verbose:
                    console.print(f"[green]File uploaded successfully: {uploaded_pdf.id}[/green]")
                
                # Step 4: Prepare and submit to API
                progress.update(main_task, description="[blue]Preparing chat request...", advance=0.5)
                
                # Update job with pre-submission status
                job_data['mistral_api']['pre_chat_timestamp'] = datetime.now().isoformat()
                job_data['current_stage'] = 'preparing_chat_request'
                self._save_job(job_id, job_data)
                
                # Create the chat message with document and formatting instructions
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": formatting_prompt,
                            },
                            {
                                "type": "document_url",
                                "document_url": signed_url.url,
                            },
                        ],
                    }
                ]
                
                # Submit to Mistral Chat API with timeout tracking
                progress.update(main_task, description="[yellow]Waiting for Mistral Chat API response...", advance=0.5)
                
                # Update job status before API call
                job_data['current_stage'] = 'waiting_for_chat_response'
                job_data['mistral_api']['chat_request_timestamp'] = datetime.now().isoformat()
                self._save_job(job_id, job_data)
                
                # Make the API call with timeout - this is where hangs typically occur
                api_start_time = time.time()
                try:
                    with TimeoutContext(timeout_seconds=120):  # 2 minute timeout with retry logic
                        chat_response = self.client.chat.complete(
                            model=self.model,
                            messages=messages,
                        )
                    
                    api_duration = time.time() - api_start_time
                    
                    # Immediately update job with successful API response
                    job_data['current_stage'] = 'processing_response'
                    job_data['mistral_api']['api_response_duration'] = api_duration
                    self._save_job(job_id, job_data)
                    
                    if self.verbose:
                        console.print(f"[green]API response received in {api_duration:.1f}s[/green]")
                    
                except (APITimeoutError, Exception) as api_error:
                    api_duration = time.time() - api_start_time
                    
                    # Try chunking strategy for large PDFs instead of OCR fallback
                    if (isinstance(api_error, APITimeoutError) or "504" in str(api_error) or "timeout" in str(api_error).lower()):
                        
                        # For large PDFs (no page ranges), try chunking
                        if not page_ranges:
                            console.print(f"[yellow]Chat API timed out ({api_duration:.1f}s), trying chunked processing...[/yellow]")
                            
                            # Update job with chunking attempt
                            job_data['current_stage'] = 'trying_chunked_processing'
                            job_data['mistral_api']['chat_api_error'] = str(api_error)
                            job_data['mistral_api']['chat_api_duration'] = api_duration
                            job_data['mistral_api']['chunking_attempted'] = True
                            self._save_job(job_id, job_data)
                            
                            try:
                                # Process in smaller chunks to preserve formatting
                                chunked_result = self._process_pdf_chunked(original_pdf_path, formatting_prompt, job_id, job_data)
                                
                                if chunked_result['status'] == 'success':
                                    # Clean up temporary file if created
                                    if temp_file and pdf_path.exists():
                                        pdf_path.unlink()
                                    
                                    return chunked_result
                                    
                            except Exception as chunk_error:
                                console.print(f"[red]Chunked processing also failed: {chunk_error}[/red]")
                                job_data['mistral_api']['chunking_error'] = str(chunk_error)
                        
                        # If still failing and OCR fallback is enabled, use it as last resort
                        elif (self.enable_fallback and self.ocr_fallback):
                            console.print(f"[yellow]Chat API failed, trying OCR fallback (formatting may be reduced)...[/yellow]")
                            
                            job_data['current_stage'] = 'trying_ocr_fallback'
                            job_data['mistral_api']['fallback_attempted'] = True
                            self._save_job(job_id, job_data)
                            
                            try:
                                fallback_start = time.time()
                                ocr_result = self.ocr_fallback.process_pdf(str(original_pdf_path), page_ranges)
                                fallback_duration = time.time() - fallback_start
                                
                                if ocr_result['status'] == 'success':
                                    console.print(f"[green]OCR fallback succeeded in {fallback_duration:.1f}s (formatting reduced)[/green]")
                                    processing_time = time.time() - start_time
                                    result = {
                                        'status': 'success',
                                        'input_file': str(original_pdf_path),
                                        'markdown_content': ocr_result['markdown_content'],
                                        'processing_time': processing_time,
                                        'model': f"{self.model} (OCR fallback - formatting reduced)",
                                        'formatting_prompt_length': len(formatting_prompt) if formatting_prompt else 0,
                                        'page_ranges': page_ranges,
                                        'job_id': job_id,
                                        'timestamp': datetime.now().isoformat(),
                                        'fallback_used': True,
                                        'formatting_preserved': False  # Important warning
                                    }
                                    
                                    job_data['status'] = 'completed'
                                    job_data['end_time'] = datetime.now().isoformat()
                                    job_data['processing_time'] = processing_time
                                    job_data['model'] = result['model']
                                    job_data['mistral_api']['fallback_success'] = True
                                    job_data['mistral_api']['fallback_duration'] = fallback_duration
                                    self._save_job(job_id, job_data)
                                    
                                    # Clean up temporary file
                                    if temp_file and pdf_path.exists():
                                        pdf_path.unlink()
                                    
                                    self._display_results(result, processing_time, len(ocr_result['markdown_content']))
                                    return result
                                    
                            except Exception as fallback_error:
                                console.print(f"[red]OCR fallback also failed: {fallback_error}[/red]")
                                job_data['mistral_api']['fallback_error'] = str(fallback_error)
                    
                    # Original error handling if no fallback or fallback failed
                    if isinstance(api_error, APITimeoutError):
                        job_data['current_stage'] = 'api_timeout'
                        job_data['mistral_api']['api_timeout'] = str(api_error)
                        job_data['mistral_api']['api_timeout_duration'] = api_duration
                        console.print(f"[red]API call timed out after {api_duration:.1f}s[/red]")
                    else:
                        job_data['current_stage'] = 'api_error'
                        job_data['mistral_api']['api_error'] = str(api_error)
                        job_data['mistral_api']['api_error_duration'] = api_duration
                        console.print(f"[red]API error after {api_duration:.1f}s: {api_error}[/red]")
                    
                    self._save_job(job_id, job_data)
                    raise api_error
                
                # Update job with chat response details
                job_data['mistral_api'].update({
                    'chat_response_id': getattr(chat_response, 'id', None),
                    'chat_model': chat_response.model,
                    'chat_timestamp': datetime.now().isoformat(),
                    'usage': {
                        'prompt_tokens': getattr(chat_response.usage, 'prompt_tokens', None) if hasattr(chat_response, 'usage') else None,
                        'completion_tokens': getattr(chat_response.usage, 'completion_tokens', None) if hasattr(chat_response, 'usage') else None,
                        'total_tokens': getattr(chat_response.usage, 'total_tokens', None) if hasattr(chat_response, 'usage') else None
                    }
                })
                self._save_job(job_id, job_data)
                
                # Step 5: Extract and finalize results
                progress.update(main_task, description="[green]Finalizing results...", advance=1)
                
                # Extract markdown content from response
                markdown_content = chat_response.choices[0].message.content
                
                processing_time = time.time() - start_time
                
                result = {
                    'status': 'success',
                    'input_file': str(original_pdf_path),
                    'markdown_content': markdown_content,
                    'processing_time': processing_time,
                    'model': self.model,
                    'formatting_prompt_length': len(formatting_prompt),
                    'page_ranges': page_ranges,
                    'job_id': job_id,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Update job status to completed with final details
                job_data['status'] = 'completed'
                job_data['end_time'] = datetime.now().isoformat()
                job_data['processing_time'] = processing_time
                job_data['markdown_length'] = len(markdown_content)
                job_data['model'] = self.model
                job_data['formatting_prompt_length'] = len(formatting_prompt)
                
                # Add complete API interaction summary
                job_data['mistral_api'].update({
                    'completion_timestamp': datetime.now().isoformat(),
                    'response_content_length': len(markdown_content),
                    'formatting_prompt_length': len(formatting_prompt)
                })
                
                self._save_job(job_id, job_data)
                
                # Clean up temporary file if created
                if temp_file and pdf_path.exists():
                    pdf_path.unlink()
                
                # Display results with Rich panel
                self._display_results(result, processing_time, len(markdown_content))
                
                return result
                
            except Exception as e:
                # Update job with error and preserve any API details captured
                job_data['status'] = 'failed'
                job_data['error'] = str(e)
                job_data['error_type'] = type(e).__name__
                job_data['end_time'] = datetime.now().isoformat()
                
                # Add error timestamp to API details if they exist
                if 'mistral_api' in job_data:
                    job_data['mistral_api']['error_timestamp'] = datetime.now().isoformat()
                
                self._save_job(job_id, job_data)
                
                # Clean up temporary file if created
                if 'temp_file' in locals() and temp_file and pdf_path.exists():
                    pdf_path.unlink()
                    
                error_result = {
                    'status': 'error',
                    'input_file': str(original_pdf_path if 'original_pdf_path' in locals() else pdf_path),
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'page_ranges': page_ranges,
                    'job_id': job_id,
                    'timestamp': datetime.now().isoformat()
                }
                
                if self.verbose:
                    console.print(f"[red]Chat OCR failed: {e}[/red]")
                
                return error_result
    
    def _process_pdf_chunked(self, pdf_path: Path, formatting_prompt: str, job_id: str, job_data: dict, progress=None, main_task=None) -> Dict[str, Any]:
        """
        Process large PDF by breaking it into smaller chunks to preserve Chat API formatting.
        
        Args:
            pdf_path: Path to PDF file
            formatting_prompt: Formatting instructions
            job_id: Job identifier
            job_data: Job tracking data
            
        Returns:
            Combined result from all chunks
        """
        start_time = time.time()
        
        # Determine PDF size and optimal chunk size
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        doc.close()
        
        # Dynamic chunk size based on total pages (smaller chunks for large docs)
        if total_pages <= 10:
            chunk_size = 5
        elif total_pages <= 30:
            chunk_size = 10  
        else:
            chunk_size = 15
        
        console.print(f"[cyan]Processing {total_pages} pages in chunks of {chunk_size}[/cyan]")
        
        chunks = []
        chunk_results = []
        
        # Create page range chunks
        for start_page in range(1, total_pages + 1, chunk_size):
            end_page = min(start_page + chunk_size - 1, total_pages)
            chunk_range = f"{start_page}-{end_page}" if start_page != end_page else str(start_page)
            chunks.append(chunk_range)
        
        # Setup working file for incremental saving
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        working_file = output_dir / f"{pdf_path.stem}_{job_id}_WORKING.md"
        
        # Use existing progress bar if available, otherwise create a new one
        if progress and main_task:
            # Update existing progress bar for chunk processing
            chunk_task = main_task
            use_existing_progress = True
        else:
            # Create new progress bar if not passed
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console
            )
            chunk_task = progress.add_task(f"[cyan]Processing {len(chunks)} chunks", total=len(chunks))
            use_existing_progress = False
        
        # Process each chunk with Chat API but without verbose output to avoid conflicts
        original_verbose = self.verbose
        self.verbose = False  # Disable verbose output for chunks
        
        try:
            for i, chunk_range in enumerate(chunks):
                if use_existing_progress:
                    progress.update(chunk_task, description=f"[blue]Processing chunk {i+1}/{len(chunks)}: pages {chunk_range}")
                else:
                    progress.update(chunk_task, description=f"[blue]Processing chunk {i+1}/{len(chunks)}: pages {chunk_range}")
                
                try:
                    chunk_result = self.process_pdf_with_chat(
                        str(pdf_path), 
                        formatting_prompt, 
                        chunk_range,
                        skip_optimization=True,  # Already optimized
                        keep_images=False,
                        chunk_size=15,
                        chunk_threshold=20,
                        disable_chunking=True,  # Prevent recursion
                        show_progress=False  # Prevent progress bar conflicts
                    )
                    
                    if chunk_result['status'] == 'success':
                        chunk_content = chunk_result['markdown_content']
                        chunk_results.append({
                            'pages': chunk_range,
                            'content': chunk_content,
                            'processing_time': chunk_result['processing_time']
                        })
                        console.print(f"[green]Chunk {i+1} completed ({chunk_result['processing_time']:.1f}s)[/green]")
                    else:
                        chunk_content = f"<!-- ERROR: Chunk {chunk_range} failed: {chunk_result.get('error', 'Unknown error')} -->"
                        console.print(f"[red]Chunk {i+1} failed: {chunk_result.get('error', 'Unknown error')}[/red]")
                        chunk_results.append({
                            'pages': chunk_range,
                            'content': chunk_content,
                            'processing_time': 0
                        })
                    
                    # Save incremental progress to working file
                    incremental_content = []
                    for result in chunk_results:
                        incremental_content.append(f"<!-- Pages {result['pages']} -->\n{result['content']}\n")
                    
                    with open(working_file, 'w', encoding='utf-8') as f:
                        f.write("\n".join(incremental_content))
                    
                    # Small delay between chunks to avoid rate limiting
                    time.sleep(2)
                    
                except Exception as chunk_error:
                    console.print(f"[red]Chunk {i+1} error: {chunk_error}[/red]")
                    chunk_content = f"<!-- ERROR: Chunk {chunk_range} error: {chunk_error} -->"
                    chunk_results.append({
                        'pages': chunk_range,
                        'content': chunk_content,
                        'processing_time': 0
                    })
                    
                    # Save incremental progress even on error
                    incremental_content = []
                    for result in chunk_results:
                        incremental_content.append(f"<!-- Pages {result['pages']} -->\n{result['content']}\n")
                    
                    with open(working_file, 'w', encoding='utf-8') as f:
                        f.write("\n".join(incremental_content))
                
                # Update progress (only advance if we have our own progress bar)
                if not use_existing_progress:
                    progress.advance(chunk_task)
            
        finally:
            # Restore original verbose setting
            self.verbose = original_verbose
        
        # Combine all chunk results
        combined_content = []
        total_chunk_time = 0
        successful_chunks = 0
        
        for chunk_result in chunk_results:
            if not chunk_result['content'].startswith('<!-- ERROR'):
                successful_chunks += 1
            combined_content.append(f"<!-- Pages {chunk_result['pages']} -->\n{chunk_result['content']}\n")
            total_chunk_time += chunk_result['processing_time']
        
        final_content = "\n".join(combined_content)
        processing_time = time.time() - start_time
        
        # Create final result
        result = {
            'status': 'success' if successful_chunks > 0 else 'error',
            'input_file': str(pdf_path),
            'markdown_content': final_content,
            'processing_time': processing_time,
            'model': f"{self.model} (chunked: {successful_chunks}/{len(chunks)} chunks)",
            'formatting_prompt_length': len(formatting_prompt),
            'page_ranges': f"all ({len(chunks)} chunks)",
            'job_id': job_id,
            'timestamp': datetime.now().isoformat(),
            'chunked': True,
            'chunks_processed': successful_chunks,
            'total_chunks': len(chunks),
            'chunk_processing_time': total_chunk_time
        }
        
        # Update job data
        job_data['status'] = 'completed' if successful_chunks > 0 else 'failed'
        job_data['current_stage'] = 'completed_chunked'
        job_data['end_time'] = datetime.now().isoformat()
        job_data['processing_time'] = processing_time
        job_data['markdown_length'] = len(final_content)
        job_data['model'] = result['model']
        
        # Initialize mistral_api section if it doesn't exist
        if 'mistral_api' not in job_data:
            job_data['mistral_api'] = {}
        
        job_data['mistral_api']['chunking_success'] = True
        job_data['mistral_api']['chunks_processed'] = successful_chunks
        job_data['mistral_api']['total_chunks'] = len(chunks)
        job_data['mistral_api']['chunk_processing_time'] = total_chunk_time
        self._save_job(job_id, job_data)
        
        console.print(f"[green]Chunked processing completed: {successful_chunks}/{len(chunks)} chunks successful[/green]")
        
        # Rename working file to final output file if successful
        if successful_chunks > 0:
            final_output_file = output_dir / f"{pdf_path.stem}_laser_focused_{job_id}.md"
            try:
                working_file.rename(final_output_file)
                console.print(f"[green]Final output saved to: {final_output_file}[/green]")
                result['output_file'] = str(final_output_file)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not rename working file: {e}[/yellow]")
                console.print(f"[yellow]Working file remains at: {working_file}[/yellow]")
                result['output_file'] = str(working_file)
        else:
            console.print(f"[yellow]Processing failed, partial results saved to: {working_file}[/yellow]")
            result['output_file'] = str(working_file)
        
        # Display results
        self._display_results(result, processing_time, len(final_content))
        
        return result
    
    def _extract_pages(self, pdf_path: Path, page_ranges: str) -> Path:
        """
        Extract specific pages from PDF and create a temporary file.
        
        Args:
            pdf_path: Path to original PDF
            page_ranges: Page ranges string (e.g., "1-3,5,7-9")
            
        Returns:
            Path to temporary PDF with extracted pages
        """
        pages_to_extract = self._parse_page_ranges(page_ranges)
        
        if self.verbose:
            console.print(f"[blue]Extracting pages {page_ranges} ({len(pages_to_extract)} pages)[/blue]")
        
        # Open source PDF
        src_doc = fitz.open(pdf_path)
        
        # Create new PDF with selected pages
        new_doc = fitz.open()
        
        for page_num in pages_to_extract:
            # Convert to 0-indexed for PyMuPDF
            page_index = page_num - 1
            if 0 <= page_index < src_doc.page_count:
                new_doc.insert_pdf(src_doc, from_page=page_index, to_page=page_index)
            else:
                console.print(f"[yellow]Warning: Page {page_num} out of range (1-{src_doc.page_count})[/yellow]")
        
        # Save to temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf', prefix='mistral_chat_')
        os.close(temp_fd)
        
        new_doc.save(temp_path)
        new_doc.close()
        src_doc.close()
        
        if self.verbose:
            console.print(f"[green]Created temporary PDF: {temp_path}[/green]")
        
        return Path(temp_path)
    
    def _parse_page_ranges(self, page_ranges: str) -> List[int]:
        """
        Parse page ranges string into list of page numbers.
        
        Args:
            page_ranges: String like "1-3,5,7-9"
            
        Returns:
            List of page numbers (1-indexed)
        """
        pages = []
        
        for part in page_ranges.replace(' ', '').split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                pages.extend(range(start, end + 1))
            else:
                pages.append(int(part))
        
        return sorted(set(pages))
    
    def _get_pdf_page_count(self, pdf_path: Path) -> int:
        """
        Get the total page count of a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Number of pages in the PDF
        """
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        doc.close()
        return page_count
    
    def _get_default_formatting_prompt(self) -> str:
        """
        Get the default formatting preservation prompt (streamlined for Chat API).
        
        Returns:
            Concise prompt for formatting preservation
        """
        return """Convert this PDF to Markdown. Preserve ALL formatting:
- **Bold text** → **bold** markdown
- *Italic text* → *italic* markdown  
- ***Bold+italic*** → ***bold+italic*** markdown
- Headers → # ## ### hierarchy
- Lists and tables preserved
- Scan every word for font styling
Output only markdown, no commentary."""
    
    def process_with_custom_prompt(self, pdf_path: str, custom_prompt: str, 
                                 page_ranges: Optional[str] = None, skip_optimization: bool = False, 
                                 keep_images: bool = False, chunk_size: int = 15, 
                                 chunk_threshold: int = 20, disable_chunking: bool = False) -> Dict[str, Any]:
        """
        Process PDF with a completely custom prompt.
        
        Args:
            pdf_path: Path to PDF file
            custom_prompt: Custom prompt for processing
            page_ranges: Optional page ranges (e.g., "1-3,5,7-9")
            
        Returns:
            Dict with processing results
        """
        return self.process_pdf_with_chat(pdf_path, custom_prompt, page_ranges, skip_optimization, keep_images, chunk_size, chunk_threshold, disable_chunking)
    
    def test_formatting_approaches(self, pdf_path: str, page_ranges: Optional[str] = None, skip_optimization: bool = False, keep_images: bool = False, chunk_size: int = 15, chunk_threshold: int = 20, disable_chunking: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Test multiple formatting approaches on the same document.
        
        Args:
            pdf_path: Path to PDF file
            page_ranges: Optional page ranges (e.g., "1-3,5,7-9")
            
        Returns:
            Dict with results from different approaches
        """
        approaches = {
            'default': self._get_default_formatting_prompt(),
            'aggressive': self._get_aggressive_formatting_prompt(),
            'minimal': self._get_minimal_formatting_prompt(),
            'llamaparse': self._get_llamaparse_formatting_prompt(),
            'italic_focused': self._get_italic_focused_prompt(),
            'style_hunter': self._get_style_hunter_prompt(),
            'ultra_precise': self._get_ultra_precise_prompt(),
            'laser_focused': self._get_laser_focused_prompt(),
        }
        
        results = {}
        
        for approach_name, prompt in approaches.items():
            if self.verbose:
                console.print(f"[cyan]Testing approach: {approach_name}[/cyan]")
            
            result = self.process_pdf_with_chat(pdf_path, prompt, page_ranges, skip_optimization, keep_images, chunk_size, chunk_threshold, disable_chunking)
            results[approach_name] = result
            
            # Small delay between requests
            time.sleep(1)
        
        return results
    
    def _get_aggressive_formatting_prompt(self) -> str:
        """Aggressive formatting preservation prompt."""
        return """
URGENT FORMATTING PRESERVATION TASK:

You are a formatting preservation specialist. Convert this PDF to markdown with ZERO tolerance for lost formatting.

MANDATORY REQUIREMENTS:
- Every single bold word becomes **bold**
- Every single italic word becomes *italic*  
- Every emphasized phrase gets proper markdown
- ALL CAPS sections likely indicate emphasis - make them **bold**
- Underlined text becomes **bold** (markdown limitation)
- Section headers must use # ## ### hierarchy
- Preserve ALL visual emphasis without exception

SCANNING PROTOCOL:
1. Examine each word for font weight/style changes
2. Look for visual emphasis patterns (spacing, capitalization)
3. Identify hierarchical structure (headers, subheaders)
4. Preserve list formatting and indentation
5. Convert tables to proper markdown tables

FAILURE TO PRESERVE FORMATTING IS UNACCEPTABLE.

Convert the document now, preserving every bit of visual emphasis.
"""
    
    def _get_minimal_formatting_prompt(self) -> str:
        """Minimal formatting prompt for comparison."""
        return """
Convert this PDF document to clean markdown format. 
Preserve basic structure including headers, lists, and tables.
Make sure bold and italic text are converted to proper markdown syntax.
"""
    
    def _get_llamaparse_formatting_prompt(self) -> str:
        """LlamaParse-proven formatting prompt that works for style preservation."""
        return """
MANDATORY FORMATTING REQUIREMENTS: You MUST NEVER ignore text formatting. ALWAYS convert bold text to **bold** markdown and italic text to *italic* markdown. FAILURE TO PRESERVE FORMATTING IS UNACCEPTABLE. Scan every single word for font weight changes, emphasis, and styling. Bold headings, author names, game titles, and emphasized terms are CRITICAL and must be preserved. This is a STRICT REQUIREMENT - do not skip any formatted text.

Convert this PDF document to markdown format with the above formatting requirements strictly enforced.
"""
    
    def _get_italic_focused_prompt(self) -> str:
        """Enhanced prompt specifically targeting italic and combined formatting detection."""
        return """
CRITICAL ITALIC AND FORMATTING DETECTION PROTOCOL:

You are an expert typography analyzer. Your mission is to detect and preserve ALL text styling with surgical precision.

MANDATORY DETECTION REQUIREMENTS:
1. **ITALIC TEXT**: Any slanted, angled, or stylistically different text becomes *italic*
2. **BOLD TEXT**: Any thick, heavy, or emphasized text becomes **bold**
3. **BOLD+ITALIC**: Text that is both thick AND slanted becomes ***bold+italic***
4. **UNDERLINED**: Convert to **bold** (markdown limitation)
5. **ALL CAPS**: Often indicates emphasis - make **bold** unless clearly just capitalization

VISUAL ANALYSIS PROTOCOL:
- Examine EVERY word for font weight differences (thickness variations)
- Look for font slant/angle changes (italic indicators) 
- Detect font style shifts within sentences
- Identify emphasis patterns: spacing, sizing, styling
- Note any visual emphasis markers: bullets, dashes, special characters

SPECIFIC TEXT PATTERNS TO WATCH:
- Game terms, rules, mechanics (often italicized)
- Move names, ability names (often bold or bold+italic)
- Flavor text, quotes (often italic)
- Section headers (often bold)
- Important instructions (often bold or italic)
- Character names, place names (often italic)

FORMATTING SYNTAX:
- Single emphasis: *italic text*
- Strong emphasis: **bold text**  
- Combined emphasis: ***bold and italic text***

ZERO TOLERANCE POLICY: Do not miss ANY styled text. Every font change must be captured.

Convert the PDF document now with complete style preservation.
"""
    
    def _get_style_hunter_prompt(self) -> str:
        """Aggressive style hunting prompt using multiple detection strategies."""
        return """
STYLE HUNTER MODE ACTIVATED - MAXIMUM FORMATTING PRESERVATION

You are a forensic typography expert. Hunt down every single style variation in this document.

MULTI-LAYER DETECTION STRATEGY:

LAYER 1 - VISUAL WEIGHT ANALYSIS:
- Scan for ANY thickness variations between words
- Thick/heavy text = **bold**
- Normal weight = no formatting
- Light text = check if it should be emphasized differently

LAYER 2 - SLANT/ANGLE DETECTION:
- Look for ANY text that appears slanted or angled
- Slanted text = *italic*
- Forward-leaning characters = *italic*
- Stylistic font variations = *italic*

LAYER 3 - COMBINED STYLE DETECTION:
- Text that is BOTH thick AND slanted = ***bold+italic***
- Text that is thick with underlines = **bold**
- Text in special fonts or styles = appropriate markdown

LAYER 4 - CONTEXTUAL EMPHASIS DETECTION:
- Words set apart by spacing = likely *italic* or **bold**
- Terms in quotation marks = often *italic*
- Technical terms = often *italic*
- Important warnings/notes = often **bold**
- Game mechanics = mix of **bold** and *italic*

LAYER 5 - SEMANTIC FORMATTING HINTS:
- Character abilities/moves = **bold** or ***bold+italic***
- Flavor text/descriptions = *italic*
- Rules/mechanics = **bold**
- Examples = *italic*
- Warnings/important notes = **bold**

CRITICAL PATTERNS FOR TTRPG DOCUMENTS:
- Move names: "Sixth sense", "Battlefield grace" = likely **bold**
- Game terms: "roll+weird", "Hx", stat names = likely **bold**
- Descriptive text: background flavor = likely *italic*
- Instructions: "when you...", "on a hit..." = check for **bold**

OUTPUT REQUIREMENTS:
- Use *single asterisks* for italic
- Use **double asterisks** for bold  
- Use ***triple asterisks*** for bold+italic
- Preserve ALL detected styling without exception

Execute comprehensive style detection now.
"""
    
    def _get_ultra_precise_prompt(self) -> str:
        """Ultra-aggressive prompt targeting specific formatting patterns known to exist."""
        return """
CRITICAL MISSION: DETECT SUBTLE ITALIC AND BOLD-ITALIC FORMATTING

You are a master typography forensics expert. This PDF contains specific formatting that MUST be detected:

TARGET FORMATTING EXAMPLES THAT EXIST IN THIS DOCUMENT:
1. "stabilize and heal someone at 9:00 or past" = ***BOLD-ITALIC*** (thick AND slanted)
2. "Visions of death" = ***BOLD-ITALIC*** (thick AND slanted)  
3. Long descriptive passages like "a night in high luxury & company; any weapon, gear..." = *ITALIC* (slanted only)

ULTRA-PRECISE DETECTION PROTOCOL:

PHASE 1 - FONT WEIGHT DETECTION:
- THICK/HEAVY text = **bold**
- NORMAL weight text = no formatting
- Examine every single word for thickness variations

PHASE 2 - FONT SLANT DETECTION: 
- ANY slanted/angled text = *italic*
- Forward-leaning characters = *italic*
- Look for subtle angle differences from vertical text
- Even slight slants must be detected

PHASE 3 - COMBINED STYLE DETECTION:
- Text that is BOTH thick AND slanted = ***bold-italic***
- This is CRITICAL - many important terms use bold-italic
- Look especially at: move names, special terms, emphasized phrases

PHASE 4 - CONTEXTUAL CLUES:
- Game move names often bold-italic: "Visions of death", "stabilize and heal"
- Long descriptive text often italic: equipment lists, flavor text
- Technical terms often italic: "hi-tech", "bodyguard"
- Rules text often bold-italic when emphasized

MANDATORY MARKDOWN OUTPUT:
- Slanted text only: *italic text*
- Thick text only: **bold text**  
- Thick AND slanted: ***bold-italic text***

FAILURE CRITERIA: Missing ANY slanted text is unacceptable failure.

SCANNING INSTRUCTIONS:
1. Examine EVERY word individually for slant and weight
2. Pay special attention to phrases starting with "when you", "on a hit", ability names
3. Look for italicized equipment lists and descriptions
4. Detect bold-italic move names and special terms
5. Do not assume - verify every character's visual properties

ZERO TOLERANCE: You must detect the italic and bold-italic text that exists in this document.

Begin ultra-precise typography analysis now.
"""
    
    def _get_laser_focused_prompt(self) -> str:
        """Laser-focused prompt targeting the exact known formatting issues."""
        return """
MANDATORY FORMATTING REQUIREMENTS: You MUST NEVER ignore text formatting. ALWAYS convert bold text to **bold** markdown and italic text to *italic* markdown. FAILURE TO PRESERVE FORMATTING IS UNACCEPTABLE.

CRITICAL: This document contains these specific formats that MUST be detected:
- "stabilize and heal someone at 9:00 or past" should be ***bold-italic***
- "Visions of death" should be ***bold-italic***  
- Equipment descriptions like "a night in high luxury & company; any weapon, gear..." should be *italic*

DETECTION PRIORITY:
1. Look for slanted/angled text = *italic*
2. Look for thick text = **bold**  
3. Look for thick AND slanted text = ***bold-italic***

Scan every single word for font weight AND slant changes. Convert this PDF to markdown preserving ALL formatting.
"""
    
    def list_jobs(self, limit: int = 10):
        """List recent processing jobs"""
        jobs = []
        for job_file in sorted(self.jobs_dir.glob("*.json"), reverse=True)[:limit]:
            with open(job_file, 'r') as f:
                jobs.append(json.load(f))
        
        if not jobs:
            console.print("[yellow]No jobs found[/yellow]")
            return
        
        from rich.table import Table
        table = Table(title="Recent Processing Jobs")
        table.add_column("Job ID", style="cyan")
        table.add_column("Input File", style="white")
        table.add_column("Status", style="white")
        table.add_column("Start Time", style="white")
        
        for job in jobs:
            status_style = {
                'completed': 'green',
                'failed': 'red',
                'started': 'yellow'
            }.get(job['status'], 'white')
            
            table.add_row(
                job['id'],
                Path(job['input_file']).name,
                f"[{status_style}]{job['status']}[/{status_style}]",
                job['start_time'][:19].replace('T', ' ')
            )
        
        console.print(table)
    
    def check_job(self, job_id: str):
        """Check specific job status"""
        job_file = self.jobs_dir / f"{job_id}.json"
        
        if not job_file.exists():
            console.print(f"[red]Job not found: {job_id}[/red]")
            return
        
        with open(job_file, 'r') as f:
            job = json.load(f)
        
        from rich.panel import Panel
        # Calculate timing information
        start_time = datetime.fromisoformat(job['start_time'])
        current_time = datetime.now()
        elapsed_time = (current_time - start_time).total_seconds()
        
        # Build panel content with current stage info
        panel_content = f"""[bold]Job ID:[/bold] {job['id']}
[bold]Status:[/bold] {job['status']}
[bold]Input:[/bold] {job['input_file']}
[bold]Started:[/bold] {job['start_time']}
[bold]Elapsed Time:[/bold] {elapsed_time:.1f}s
[bold]Pages:[/bold] {job.get('page_ranges', 'all')}
[bold]Prompt Type:[/bold] {job.get('formatting_prompt_type', 'unknown')}"""

        # Add current stage information for in-progress jobs
        if job['status'] == 'started' and 'current_stage' in job:
            stage = job['current_stage']
            panel_content += f"\n[bold]Current Stage:[/bold] {stage}"
            
            # Add stage-specific timing
            if 'mistral_api' in job:
                api_data = job['mistral_api']
                if stage == 'waiting_for_chat_response' and 'chat_request_timestamp' in api_data:
                    request_time = datetime.fromisoformat(api_data['chat_request_timestamp'])
                    wait_duration = (current_time - request_time).total_seconds()
                    panel_content += f"\n[bold]API Wait Time:[/bold] {wait_duration:.1f}s"
                    
                    if wait_duration > 120:  # Over 2 minutes
                        panel_content += f"\n[red]⚠️  Long API wait detected[/red]"
        
        # Color the border based on status and timing
        border_color = "cyan"
        if job['status'] == 'started' and elapsed_time > 300:  # Over 5 minutes
            border_color = "yellow"
        elif job['status'] == 'started' and elapsed_time > 600:  # Over 10 minutes
            border_color = "red"
            
        panel = Panel.fit(
            panel_content,
            title=f"Job Details - {job_id}",
            border_style=border_color
        )
        
        console.print(panel)
        
        if job['status'] == 'completed':
            console.print("\n[green]Processing Results:[/green]")
            console.print(f"  Processing time: {job.get('processing_time', 'N/A'):.1f}s")
            console.print(f"  Markdown length: {job.get('markdown_length', 'N/A')} characters")
            console.print(f"  Model used: {job.get('model', 'N/A')}")
            
            # Display Mistral API details if available
            if 'mistral_api' in job:
                api_data = job['mistral_api']
                console.print("\n[cyan]Mistral API Details:[/cyan]")
                console.print(f"  File upload ID: {api_data.get('file_upload_id', 'N/A')}")
                console.print(f"  Chat response ID: {api_data.get('chat_response_id', 'N/A')}")
                
                if 'usage' in api_data and api_data['usage']['total_tokens']:
                    usage = api_data['usage']
                    console.print(f"  Token usage: {usage['total_tokens']} total ({usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion)")
                    
        elif job['status'] == 'failed' and 'error' in job:
            console.print(f"\n[red]Error: {job['error']}[/red]")
            
            # Display API details if available even for failed jobs
            if 'mistral_api' in job:
                api_data = job['mistral_api']
                console.print("\n[yellow]Mistral API Details (before failure):[/yellow]")
                if 'file_upload_id' in api_data:
                    console.print(f"  File upload ID: {api_data['file_upload_id']}")
                if 'chat_response_id' in api_data:
                    console.print(f"  Chat response ID: {api_data['chat_response_id']}")
                if 'usage' in api_data and api_data['usage']['total_tokens']:
                    usage = api_data['usage']
                    console.print(f"  Token usage: {usage['total_tokens']} total")
    
    def check_hung_jobs(self):
        """Check for jobs that may be hung or stuck"""
        jobs = []
        current_time = datetime.now()
        
        for job_file in self.jobs_dir.glob("*.json"):
            with open(job_file, 'r') as f:
                job = json.load(f)
                
            # Only check jobs that are still "started"
            if job['status'] == 'started':
                start_time = datetime.fromisoformat(job['start_time'])
                elapsed_time = (current_time - start_time).total_seconds()
                
                # Consider jobs hung if they've been running for more than 10 minutes
                if elapsed_time > 600:
                    job['elapsed_time'] = elapsed_time
                    jobs.append(job)
        
        if not jobs:
            console.print("[green]No hung jobs detected[/green]")
            return
            
        console.print(f"[yellow]Found {len(jobs)} potentially hung job(s):[/yellow]\n")
        
        for job in jobs:
            console.print(f"[red]🚨 Job: {job['id']}[/red]")
            console.print(f"   Started: {job['start_time']}")
            console.print(f"   Elapsed: {job['elapsed_time']:.1f}s ({job['elapsed_time']/60:.1f} minutes)")
            console.print(f"   Stage: {job.get('current_stage', 'unknown')}")
            
            # Show API wait time if stuck in API call
            if 'mistral_api' in job and 'chat_request_timestamp' in job['mistral_api']:
                request_time = datetime.fromisoformat(job['mistral_api']['chat_request_timestamp'])
                api_wait_time = (current_time - request_time).total_seconds()
                console.print(f"   API Wait: {api_wait_time:.1f}s")
                
            console.print(f"   File: {job['input_file']}")
            console.print(f"   Use: python pdf2md_chat.py --check-job {job['id']}")
            console.print("")
    
    def _save_job(self, job_id: str, job_data: Dict[str, Any]):
        """Save job data to file"""
        job_file = self.jobs_dir / f"{job_id}.json"
        with open(job_file, 'w') as f:
            json.dump(job_data, f, indent=2, default=str)
    
    def _display_results(self, result: Dict[str, Any], processing_time: float, markdown_length: int):
        """Display conversion results with Rich panel"""
        panel = Panel.fit(
            f"""[bold green]✓ Conversion completed successfully![/bold green]
            
[bold]Job ID:[/bold] {result['job_id']}
[bold]Input:[/bold] {Path(result['input_file']).name}
[bold]Pages:[/bold] {result['page_ranges'] or 'all'}
[bold]Processing time:[/bold] {processing_time:.1f}s
[bold]Markdown length:[/bold] {markdown_length} characters
[bold]Model:[/bold] {result['model']}
[bold]Prompt length:[/bold] {result['formatting_prompt_length']} characters""",
            title="Conversion Results",
            border_style="green"
        )
        
        console.print("\n", panel)


def main():
    """Main function for testing Mistral Chat OCR."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Mistral Chat OCR for formatting preservation")
    parser.add_argument("pdf_path", nargs="?", help="Path to PDF file")
    parser.add_argument("--approach", choices=['default', 'aggressive', 'minimal', 'llamaparse', 'italic_focused', 'style_hunter', 'ultra_precise', 'laser_focused', 'all'], 
                       default='default', help="Formatting approach to test")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--model", default="mistral-small-latest", help="Mistral model to use")
    parser.add_argument("--custom-prompt", help="Use a custom prompt from file")
    parser.add_argument("--pages", help="Page ranges to process (e.g., '1-3,5,7-9')")
    
    # Job management
    parser.add_argument("--list-jobs", action="store_true", help="List recent processing jobs")
    parser.add_argument("--check-job", help="Check specific job status")
    parser.add_argument("--check-hung", action="store_true", help="Check for hung/stuck jobs")
    
    # Optimization options
    parser.add_argument("--no-optimize", action="store_true", help="Skip PDF optimization step")
    parser.add_argument("--keep-images", action="store_true", help="Keep images during optimization")
    
    # Chunking options
    parser.add_argument("--chunk-size", type=int, default=15, help="Pages per chunk for large documents (default: 15)")
    parser.add_argument("--chunk-threshold", type=int, default=20, help="Page count threshold to trigger chunking (default: 20)")
    parser.add_argument("--no-chunk", action="store_true", help="Disable automatic chunking (process entire document)")
    
    # API options
    parser.add_argument("--no-fallback", action="store_true", help="Disable automatic OCR API fallback on Chat API failures")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize Mistral Chat OCR
        chat_ocr = MistralChatOCR(model=args.model, verbose=True, enable_fallback=not args.no_fallback)
        
        # Handle job management commands first
        if args.list_jobs:
            chat_ocr.list_jobs()
            return
        elif args.check_job:
            chat_ocr.check_job(args.check_job)
            return
        elif args.check_hung:
            chat_ocr.check_hung_jobs()
            return
        
        # Check if PDF file was provided for processing
        if not args.pdf_path:
            parser.print_help()
            return
            
        # Generate timestamp for output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_name = Path(args.pdf_path).stem
        
        if args.custom_prompt:
            # Use custom prompt from file
            with open(args.custom_prompt, 'r') as f:
                custom_prompt = f.read()
            
            console.print(f"[blue]Using custom prompt from: {args.custom_prompt}[/blue]")
            result = chat_ocr.process_with_custom_prompt(args.pdf_path, custom_prompt, args.pages, 
                                                       skip_optimization=args.no_optimize, keep_images=args.keep_images,
                                                       chunk_size=args.chunk_size, chunk_threshold=args.chunk_threshold, 
                                                       disable_chunking=args.no_chunk)
            
            # Handle results with comprehensive error handling
            job_id = result.get('job_id', f'job_{timestamp}')
            output_file = output_dir / f"{pdf_name}_custom_{job_id}.md"
            metadata_file = output_dir / f"{pdf_name}_custom_{job_id}_metadata.json"
            
            # Create comprehensive metadata for both success and failure
            comprehensive_metadata = {
                'source_pdf': Path(args.pdf_path).name,
                'output_markdown': output_file.name if result['status'] == 'success' else None,
                'processing_date': datetime.now().isoformat(),
                'job_id': job_id,
                'pages_processed': args.pages or "all",
                'approach': 'custom',
                'custom_prompt_file': args.custom_prompt,
                'status': result['status'],
                'processing_time': {
                    'total': result.get('processing_time', 0)
                },
                'mistral_model': result.get('model', 'unknown'),
                'formatting_prompt_length': result.get('formatting_prompt_length', 0),
                'markdown_length': len(result.get('markdown_content', '')) if result['status'] == 'success' else 0,
                'raw_result': result
            }
            
            # Handle success case
            if result['status'] == 'success':
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result['markdown_content'])
                console.print(f"[green]Results saved to: {output_file}[/green]")
            
            # Handle error case with comprehensive error metadata
            else:
                comprehensive_metadata.update({
                    'error': result.get('error', 'Unknown error'),
                    'error_type': result.get('error_type', 'UnknownError'),
                    'failed_at': datetime.now().isoformat()
                })
                console.print(f"[red]Processing failed: {result.get('error', 'Unknown error')}[/red]")
                console.print(f"[yellow]Error metadata saved to: {metadata_file}[/yellow]")
            
            # Always save metadata (success or failure)
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_metadata, f, indent=2, default=str)
            
            # Exit with error code if processing failed
            if result['status'] != 'success':
                sys.exit(1)
            
        elif args.approach == 'all':
            # Test all approaches
            console.print("[blue]Testing all formatting approaches...[/blue]")
            results = chat_ocr.test_formatting_approaches(args.pdf_path, args.pages, skip_optimization=args.no_optimize, keep_images=args.keep_images, chunk_size=args.chunk_size, chunk_threshold=args.chunk_threshold, disable_chunking=args.no_chunk)
            
            # Save results for each approach with comprehensive error handling
            failed_approaches = []
            for approach_name, result in results.items():
                job_id = result.get('job_id', f'job_{timestamp}_{approach_name}')
                output_file = output_dir / f"{pdf_name}_{approach_name}_{job_id}.md"
                metadata_file = output_dir / f"{pdf_name}_{approach_name}_{job_id}_metadata.json"
                
                # Create comprehensive metadata for both success and failure
                comprehensive_metadata = {
                    'source_pdf': Path(args.pdf_path).name,
                    'output_markdown': output_file.name if result['status'] == 'success' else None,
                    'processing_date': datetime.now().isoformat(),
                    'job_id': job_id,
                    'pages_processed': args.pages or "all",
                    'approach': approach_name,
                    'status': result['status'],
                    'processing_time': {
                        'total': result.get('processing_time', 0)
                    },
                    'mistral_model': result.get('model', 'unknown'),
                    'formatting_prompt_length': result.get('formatting_prompt_length', 0),
                    'markdown_length': len(result.get('markdown_content', '')) if result['status'] == 'success' else 0,
                    'raw_result': result
                }
                
                # Handle success case
                if result['status'] == 'success':
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result['markdown_content'])
                    console.print(f"[green]{approach_name} results saved to: {output_file}[/green]")
                
                # Handle error case
                else:
                    comprehensive_metadata.update({
                        'error': result.get('error', 'Unknown error'),
                        'error_type': result.get('error_type', 'UnknownError'),
                        'failed_at': datetime.now().isoformat()
                    })
                    failed_approaches.append(approach_name)
                    console.print(f"[red]{approach_name} failed: {result.get('error', 'Unknown error')}[/red]")
                    console.print(f"[yellow]Error metadata saved to: {metadata_file}[/yellow]")
                
                # Always save metadata (success or failure)
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(comprehensive_metadata, f, indent=2, default=str)
            
            # Report overall results
            total_approaches = len(results)
            successful_approaches = total_approaches - len(failed_approaches)
            
            console.print(f"\n[cyan]Summary: {successful_approaches}/{total_approaches} approaches completed successfully[/cyan]")
            if failed_approaches:
                console.print(f"[yellow]Failed approaches: {', '.join(failed_approaches)}[/yellow]")
                # Exit with error if all approaches failed
                if len(failed_approaches) == total_approaches:
                    console.print("[red]All approaches failed![/red]")
                    sys.exit(1)
            
        else:
            # Single approach
            console.print(f"[blue]Testing {args.approach} approach...[/blue]")
            
            if args.approach == 'default':
                result = chat_ocr.process_pdf_with_chat(args.pdf_path, page_ranges=args.pages, skip_optimization=args.no_optimize, keep_images=args.keep_images, chunk_size=args.chunk_size, chunk_threshold=args.chunk_threshold, disable_chunking=args.no_chunk)
            elif args.approach == 'aggressive':
                prompt = chat_ocr._get_aggressive_formatting_prompt()
                result = chat_ocr.process_pdf_with_chat(args.pdf_path, prompt, args.pages, skip_optimization=args.no_optimize, keep_images=args.keep_images, chunk_size=args.chunk_size, chunk_threshold=args.chunk_threshold, disable_chunking=args.no_chunk)
            elif args.approach == 'minimal':
                prompt = chat_ocr._get_minimal_formatting_prompt()
                result = chat_ocr.process_pdf_with_chat(args.pdf_path, prompt, args.pages, skip_optimization=args.no_optimize, keep_images=args.keep_images, chunk_size=args.chunk_size, chunk_threshold=args.chunk_threshold, disable_chunking=args.no_chunk)
            elif args.approach == 'llamaparse':
                prompt = chat_ocr._get_llamaparse_formatting_prompt()
                result = chat_ocr.process_pdf_with_chat(args.pdf_path, prompt, args.pages, skip_optimization=args.no_optimize, keep_images=args.keep_images, chunk_size=args.chunk_size, chunk_threshold=args.chunk_threshold, disable_chunking=args.no_chunk)
            elif args.approach == 'italic_focused':
                prompt = chat_ocr._get_italic_focused_prompt()
                result = chat_ocr.process_pdf_with_chat(args.pdf_path, prompt, args.pages, skip_optimization=args.no_optimize, keep_images=args.keep_images, chunk_size=args.chunk_size, chunk_threshold=args.chunk_threshold, disable_chunking=args.no_chunk)
            elif args.approach == 'style_hunter':
                prompt = chat_ocr._get_style_hunter_prompt()
                result = chat_ocr.process_pdf_with_chat(args.pdf_path, prompt, args.pages, skip_optimization=args.no_optimize, keep_images=args.keep_images, chunk_size=args.chunk_size, chunk_threshold=args.chunk_threshold, disable_chunking=args.no_chunk)
            elif args.approach == 'ultra_precise':
                prompt = chat_ocr._get_ultra_precise_prompt()
                result = chat_ocr.process_pdf_with_chat(args.pdf_path, prompt, args.pages, skip_optimization=args.no_optimize, keep_images=args.keep_images, chunk_size=args.chunk_size, chunk_threshold=args.chunk_threshold, disable_chunking=args.no_chunk)
            elif args.approach == 'laser_focused':
                prompt = chat_ocr._get_laser_focused_prompt()
                result = chat_ocr.process_pdf_with_chat(args.pdf_path, prompt, args.pages, skip_optimization=args.no_optimize, keep_images=args.keep_images, chunk_size=args.chunk_size, chunk_threshold=args.chunk_threshold, disable_chunking=args.no_chunk)
            
            # Handle results with comprehensive error handling
            job_id = result.get('job_id', f'job_{timestamp}')
            output_file = output_dir / f"{pdf_name}_{args.approach}_{job_id}.md"
            metadata_file = output_dir / f"{pdf_name}_{args.approach}_{job_id}_metadata.json"
            
            # Create comprehensive metadata for both success and failure
            comprehensive_metadata = {
                'source_pdf': Path(args.pdf_path).name,
                'output_markdown': output_file.name if result['status'] == 'success' else None,
                'processing_date': datetime.now().isoformat(),
                'job_id': job_id,
                'pages_processed': args.pages or "all",
                'approach': args.approach,
                'status': result['status'],
                'processing_time': {
                    'total': result.get('processing_time', 0)
                },
                'mistral_model': result.get('model', 'unknown'),
                'formatting_prompt_length': result.get('formatting_prompt_length', 0),
                'markdown_length': len(result.get('markdown_content', '')) if result['status'] == 'success' else 0,
                'raw_result': result
            }
            
            # Handle success case
            if result['status'] == 'success':
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result['markdown_content'])
                console.print(f"[green]Results saved to: {output_file}[/green]")
            
            # Handle error case with comprehensive error metadata
            else:
                comprehensive_metadata.update({
                    'error': result.get('error', 'Unknown error'),
                    'error_type': result.get('error_type', 'UnknownError'),
                    'failed_at': datetime.now().isoformat()
                })
                console.print(f"[red]Processing failed: {result.get('error', 'Unknown error')}[/red]")
                console.print(f"[yellow]Error metadata saved to: {metadata_file}[/yellow]")
            
            # Always save metadata (success or failure)
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_metadata, f, indent=2, default=str)
            
            # Exit with error code if processing failed
            if result['status'] != 'success':
                sys.exit(1)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        # Create error metadata for unexpected exceptions
        if 'args' in locals() and args.pdf_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_name = Path(args.pdf_path).stem
            error_job_id = f"job_{timestamp}_{pdf_name}_error"
            
            error_metadata = {
                'source_pdf': Path(args.pdf_path).name,
                'processing_date': datetime.now().isoformat(),
                'job_id': error_job_id,
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__,
                'failed_at': datetime.now().isoformat(),
                'approach': getattr(args, 'approach', 'unknown'),
                'pages_processed': getattr(args, 'pages', None) or "all",
                'failure_stage': 'main_cli_execution'
            }
            
            # Try to save error metadata
            try:
                output_dir = Path(args.output_dir) if hasattr(args, 'output_dir') else Path("output")
                output_dir.mkdir(exist_ok=True)
                error_metadata_file = output_dir / f"{pdf_name}_error_{error_job_id}_metadata.json"
                
                with open(error_metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(error_metadata, f, indent=2, default=str)
                
                console.print(f"[red]Unexpected error: {e}[/red]")
                console.print(f"[yellow]Error metadata saved to: {error_metadata_file}[/yellow]")
            except Exception as save_error:
                console.print(f"[red]Error: {e}[/red]")
                console.print(f"[red]Failed to save error metadata: {save_error}[/red]")
        else:
            console.print(f"[red]Error: {e}[/red]")
        
        sys.exit(1)



if __name__ == "__main__":
    main()