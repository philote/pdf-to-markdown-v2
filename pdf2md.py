#!/usr/bin/env python3
"""
PDF to Markdown CLI Tool

Converts TTRPG PDFs to clean, formatted Markdown using Mistral OCR
with automatic optimization and progress tracking.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from dotenv import load_dotenv

from pdf_optimizer import PDFOptimizer
from mistral_ocr import MistralOCR

# Load environment variables
load_dotenv()


class PDFToMarkdownConverter:
    """Main converter class orchestrating PDF optimization and OCR"""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize converter with required components
        
        Args:
            verbose: Enable detailed output
        """
        self.verbose = verbose
        self.console = Console()
        self.optimizer = PDFOptimizer(verbose=False)  # We'll handle output
        self.ocr = None  # Initialize only when needed
        self.jobs_dir = Path(".jobs")
        self.jobs_dir.mkdir(exist_ok=True)
        
    def convert(self, pdf_path: str, page_ranges: Optional[str] = None,
                skip_optimization: bool = False, keep_images: bool = False,
                enhanced_formatting: bool = False) -> Dict[str, Any]:
        """
        Convert PDF to Markdown with progress tracking
        
        Args:
            pdf_path: Path to input PDF
            page_ranges: Optional page ranges to process
            skip_optimization: Skip PDF optimization step
            keep_images: Keep images during optimization
            enhanced_formatting: Enable enhanced formatting for TTRPG documents
            
        Returns:
            Dict with conversion results
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Initialize OCR client when needed
        if self.ocr is None:
            self.ocr = MistralOCR(verbose=False)
        
        # Initialize job tracking first to get job ID
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{pdf_path.stem}"
        
        # Generate output paths with job ID
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        markdown_path = output_dir / f"{pdf_path.stem}_{job_id}.md"
        metadata_path = output_dir / f"{pdf_path.stem}_{job_id}_metadata.json"
        job_data = {
            'id': job_id,
            'input_file': str(pdf_path),
            'status': 'started',
            'start_time': datetime.now().isoformat(),
            'page_ranges': page_ranges
        }
        
        self._save_job(job_id, job_data)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            
            # Overall task
            main_task = progress.add_task("[cyan]Converting PDF to Markdown", total=4)
            
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
                else:
                    progress.update(main_task, description="[green]Skipping optimization", advance=1)
                    job_data['optimization'] = {'skipped': True}
                
                # Step 3: Process with OCR
                progress.update(main_task, description="[blue]Processing with Mistral OCR...", advance=1)
                ocr_result = self.ocr.process_with_formatting_prompt(
                    str(optimized_path),
                    page_ranges=page_ranges,
                    aggressive_formatting=enhanced_formatting
                )
                
                if ocr_result['status'] != 'success':
                    raise Exception(f"OCR failed: {ocr_result.get('error', 'Unknown error')}")
                
                job_data['ocr_result'] = {
                    'status': ocr_result['status'],
                    'processing_time': ocr_result['processing_time'],
                    'model': ocr_result['model']
                }
                
                # Step 4: Save results
                progress.update(main_task, description="[green]Saving results...", advance=1)
                
                # Save markdown
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(ocr_result['markdown_content'])
                
                # Prepare metadata
                metadata = {
                    'source_pdf': pdf_path.name,
                    'output_markdown': markdown_path.name,
                    'processing_date': datetime.now().isoformat(),
                    'job_id': job_id,
                    'pages_processed': page_ranges or "all",
                    'optimization': {
                        'performed': optimization_result is not None,
                        'size_reduction': f"{optimization_result['size_reduction_percent']:.1f}%" if optimization_result else "N/A",
                        'images_removed': optimization_result.get('images_removed', 0) if optimization_result else 0
                    },
                    'processing_time': {
                        'optimization': optimization_result.get('processing_time', 0) if optimization_result else 0,
                        'ocr': ocr_result['processing_time'],
                        'total': time.time() - datetime.fromisoformat(job_data['start_time']).timestamp()
                    },
                    'mistral_model': ocr_result['model']
                }
                
                # Save metadata
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                
                # Update job status
                job_data['status'] = 'completed'
                job_data['end_time'] = datetime.now().isoformat()
                job_data['output_files'] = {
                    'markdown': str(markdown_path),
                    'metadata': str(metadata_path)
                }
                self._save_job(job_id, job_data)
                
                # Display results
                self._display_results(metadata, markdown_path)
                
                return {
                    'status': 'success',
                    'markdown_path': str(markdown_path),
                    'metadata_path': str(metadata_path),
                    'job_id': job_id
                }
                
            except Exception as e:
                # Update job with error
                job_data['status'] = 'failed'
                job_data['error'] = str(e)
                job_data['end_time'] = datetime.now().isoformat()
                self._save_job(job_id, job_data)
                
                self.console.print(f"\n[red]Error: {e}[/red]")
                return {
                    'status': 'error',
                    'error': str(e),
                    'job_id': job_id
                }
    
    def list_jobs(self, limit: int = 10):
        """List recent processing jobs"""
        jobs = []
        for job_file in sorted(self.jobs_dir.glob("*.json"), reverse=True)[:limit]:
            with open(job_file, 'r') as f:
                jobs.append(json.load(f))
        
        if not jobs:
            self.console.print("[yellow]No jobs found[/yellow]")
            return
        
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
        
        self.console.print(table)
    
    def check_job(self, job_id: str):
        """Check specific job status"""
        job_file = self.jobs_dir / f"{job_id}.json"
        
        if not job_file.exists():
            self.console.print(f"[red]Job not found: {job_id}[/red]")
            return
        
        with open(job_file, 'r') as f:
            job = json.load(f)
        
        panel = Panel.fit(
            f"""[bold]Job ID:[/bold] {job['id']}
[bold]Status:[/bold] {job['status']}
[bold]Input:[/bold] {job['input_file']}
[bold]Started:[/bold] {job['start_time']}
[bold]Pages:[/bold] {job.get('page_ranges', 'all')}""",
            title=f"Job Details - {job_id}",
            border_style="cyan"
        )
        
        self.console.print(panel)
        
        if job['status'] == 'completed' and 'output_files' in job:
            self.console.print("\n[green]Output files:[/green]")
            self.console.print(f"  Markdown: {job['output_files']['markdown']}")
            self.console.print(f"  Metadata: {job['output_files']['metadata']}")
        elif job['status'] == 'failed' and 'error' in job:
            self.console.print(f"\n[red]Error: {job['error']}[/red]")
    
    def _save_job(self, job_id: str, job_data: Dict[str, Any]):
        """Save job data to file"""
        job_file = self.jobs_dir / f"{job_id}.json"
        with open(job_file, 'w') as f:
            json.dump(job_data, f, indent=2)
    
    def _display_results(self, metadata: Dict[str, Any], markdown_path: Path):
        """Display conversion results"""
        panel = Panel.fit(
            f"""[bold green]✓ Conversion completed successfully![/bold green]
            
[bold]Output:[/bold] {markdown_path}
[bold]Pages:[/bold] {metadata['pages_processed']}
[bold]Processing time:[/bold] {metadata['processing_time']['total']:.1f}s
[bold]Size reduction:[/bold] {metadata['optimization']['size_reduction']}
[bold]Images removed:[/bold] {metadata['optimization']['images_removed']}""",
            title="Conversion Results",
            border_style="green"
        )
        
        self.console.print("\n", panel)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Convert PDF documents to Markdown using Mistral OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input/rulebook.pdf
  %(prog)s input/rulebook.pdf --pages 1-50,75-100
  %(prog)s input/rulebook.pdf --no-optimize
  %(prog)s input/rulebook.pdf --enhanced-formatting
  %(prog)s --list-jobs
  %(prog)s --check-job job_20240713_103045_rulebook
        """
    )
    
    # Main command
    parser.add_argument('pdf_file', nargs='?', help='PDF file to convert')
    
    # Processing options
    parser.add_argument('--pages', help='Page ranges to process (e.g., "1-10,15-20")')
    parser.add_argument('--no-optimize', action='store_true', help='Skip PDF optimization')
    parser.add_argument('--keep-images', action='store_true', help='Keep images during optimization')
    parser.add_argument('--enhanced-formatting', action='store_true', help='Enable enhanced formatting preservation for TTRPG documents')
    
    # Job management
    parser.add_argument('--list-jobs', action='store_true', help='List recent processing jobs')
    parser.add_argument('--check-job', help='Check specific job status')
    
    # Other options
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    converter = PDFToMarkdownConverter(verbose=args.verbose)
    
    try:
        if args.list_jobs:
            converter.list_jobs()
        elif args.check_job:
            converter.check_job(args.check_job)
        elif args.pdf_file:
            # Check for API key only when converting
            if not os.environ.get('MISTRAL_API_KEY'):
                rprint("[red]Error: MISTRAL_API_KEY not found in environment variables[/red]")
                rprint("Please set your API key: export MISTRAL_API_KEY='your-key-here'")
                sys.exit(1)
            
            result = converter.convert(
                args.pdf_file,
                page_ranges=args.pages,
                skip_optimization=args.no_optimize,
                keep_images=args.keep_images,
                enhanced_formatting=args.enhanced_formatting
            )
            
            if result['status'] == 'error':
                sys.exit(1)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        rprint("\n[yellow]Conversion cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()