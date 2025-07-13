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
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import fitz  # PyMuPDF
from mistralai import Mistral
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables
load_dotenv()

console = Console()


class MistralChatOCR:
    """
    Mistral Chat API integration for document processing with custom formatting prompts.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "mistral-small-latest", verbose: bool = True):
        """
        Initialize the Mistral Chat OCR client.
        
        Args:
            api_key: Mistral API key (if None, loads from environment)
            model: Model to use for chat completions
            verbose: Enable verbose output
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        
        self.client = Mistral(api_key=self.api_key)
        self.model = model
        self.verbose = verbose
        
        if self.verbose:
            console.print(f"[green]Initialized Mistral Chat OCR with model: {self.model}[/green]")
    
    def process_pdf_with_chat(self, pdf_path: str, formatting_prompt: Optional[str] = None, 
                            page_ranges: Optional[str] = None) -> Dict[str, Any]:
        """
        Process PDF using Mistral Chat API with custom formatting instructions.
        
        Args:
            pdf_path: Path to PDF file
            formatting_prompt: Custom prompt for formatting preservation
            page_ranges: Optional page ranges (e.g., "1-3,5,7-9")
            
        Returns:
            Dict with processing results
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Extract specific pages if page_ranges provided
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
        
        try:
            # Upload the PDF file first
            if self.verbose:
                console.print("[yellow]Uploading PDF file...[/yellow]")
            
            with open(pdf_path, "rb") as f:
                uploaded_pdf = self.client.files.upload(
                    file={
                        "file_name": pdf_path.name,
                        "content": f,
                    },
                    purpose="ocr"
                )
            
            # Get signed URL for the uploaded file
            signed_url = self.client.files.get_signed_url(file_id=uploaded_pdf.id)
            
            if self.verbose:
                console.print(f"[green]File uploaded successfully: {uploaded_pdf.id}[/green]")
            
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
            
            if self.verbose:
                console.print("[yellow]Submitting to Mistral Chat API...[/yellow]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Processing document...", total=None)
                
                # Submit to Mistral Chat API
                chat_response = self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                )
                
                progress.update(task, description="Complete!")
            
            # Extract markdown content from response
            markdown_content = chat_response.choices[0].message.content
            
            processing_time = time.time() - start_time
            
            result = {
                'status': 'success',
                'input_file': str(pdf_path),
                'markdown_content': markdown_content,
                'processing_time': processing_time,
                'model': self.model,
                'formatting_prompt_length': len(formatting_prompt),
                'page_ranges': page_ranges,
                'timestamp': datetime.now().isoformat()
            }
            
            # Clean up temporary file if created
            if temp_file and pdf_path.exists():
                pdf_path.unlink()
            
            if self.verbose:
                console.print(f"[green]Chat OCR completed in {processing_time:.1f} seconds[/green]")
                console.print(f"[green]Generated {len(markdown_content)} characters of markdown[/green]")
            
            return result
            
        except Exception as e:
            # Clean up temporary file if created
            if 'temp_file' in locals() and temp_file and pdf_path.exists():
                pdf_path.unlink()
                
            error_result = {
                'status': 'error',
                'input_file': str(pdf_path),
                'error': str(e),
                'error_type': type(e).__name__,
                'page_ranges': page_ranges,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.verbose:
                console.print(f"[red]Chat OCR failed: {e}[/red]")
            
            return error_result
    
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
    
    def _get_default_formatting_prompt(self) -> str:
        """
        Get the default formatting preservation prompt.
        
        Returns:
            Default prompt for formatting preservation
        """
        return """
Convert this PDF document to clean, semantic Markdown format with strict formatting preservation requirements:

CRITICAL FORMATTING REQUIREMENTS:
1. **Bold text**: Convert ALL bold text to **bold** markdown syntax
2. **Italic text**: Convert ALL italic text to *italic* markdown syntax  
3. **Headers**: Use proper # ## ### markdown hierarchy for all headings
4. **Lists**: Preserve bullet points and numbered lists with proper indentation
5. **Tables**: Convert tables to proper markdown table format
6. **Emphasis**: Look for ANY emphasized text (bold, italic, underlined) and preserve it

SPECIFIC INSTRUCTIONS:
- Scan every word for font weight changes and styling
- Bold headings, titles, and emphasized terms are CRITICAL
- Preserve the original document structure and hierarchy
- Do not skip ANY formatted text - formatting preservation is mandatory
- If text appears emphasized visually, make it bold or italic in markdown
- Pay special attention to section headers, game terms, and important phrases

Output only the converted markdown content with no additional commentary.
"""
    
    def process_with_custom_prompt(self, pdf_path: str, custom_prompt: str, 
                                 page_ranges: Optional[str] = None) -> Dict[str, Any]:
        """
        Process PDF with a completely custom prompt.
        
        Args:
            pdf_path: Path to PDF file
            custom_prompt: Custom prompt for processing
            page_ranges: Optional page ranges (e.g., "1-3,5,7-9")
            
        Returns:
            Dict with processing results
        """
        return self.process_pdf_with_chat(pdf_path, custom_prompt, page_ranges)
    
    def test_formatting_approaches(self, pdf_path: str, page_ranges: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
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
            
            result = self.process_pdf_with_chat(pdf_path, prompt, page_ranges)
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


def main():
    """Main function for testing Mistral Chat OCR."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Mistral Chat OCR for formatting preservation")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--approach", choices=['default', 'aggressive', 'minimal', 'llamaparse', 'italic_focused', 'style_hunter', 'ultra_precise', 'laser_focused', 'all'], 
                       default='default', help="Formatting approach to test")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--model", default="mistral-small-latest", help="Mistral model to use")
    parser.add_argument("--custom-prompt", help="Use a custom prompt from file")
    parser.add_argument("--pages", help="Page ranges to process (e.g., '1-3,5,7-9')")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize Mistral Chat OCR
        chat_ocr = MistralChatOCR(model=args.model, verbose=True)
        
        # Generate timestamp for output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_name = Path(args.pdf_path).stem
        
        if args.custom_prompt:
            # Use custom prompt from file
            with open(args.custom_prompt, 'r') as f:
                custom_prompt = f.read()
            
            console.print(f"[blue]Using custom prompt from: {args.custom_prompt}[/blue]")
            result = chat_ocr.process_with_custom_prompt(args.pdf_path, custom_prompt, args.pages)
            
            # Save results
            output_file = output_dir / f"{pdf_name}_custom_{timestamp}.md"
            metadata_file = output_dir / f"{pdf_name}_custom_{timestamp}_metadata.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result['markdown_content'])
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)
            
            console.print(f"[green]Results saved to: {output_file}[/green]")
            
        elif args.approach == 'all':
            # Test all approaches
            console.print("[blue]Testing all formatting approaches...[/blue]")
            results = chat_ocr.test_formatting_approaches(args.pdf_path, args.pages)
            
            # Save results for each approach
            for approach_name, result in results.items():
                output_file = output_dir / f"{pdf_name}_{approach_name}_{timestamp}.md"
                metadata_file = output_dir / f"{pdf_name}_{approach_name}_{timestamp}_metadata.json"
                
                if result['status'] == 'success':
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result['markdown_content'])
                
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, default=str)
                
                console.print(f"[green]{approach_name} results saved to: {output_file}[/green]")
            
        else:
            # Single approach
            console.print(f"[blue]Testing {args.approach} approach...[/blue]")
            
            if args.approach == 'default':
                result = chat_ocr.process_pdf_with_chat(args.pdf_path, page_ranges=args.pages)
            elif args.approach == 'aggressive':
                prompt = chat_ocr._get_aggressive_formatting_prompt()
                result = chat_ocr.process_pdf_with_chat(args.pdf_path, prompt, args.pages)
            elif args.approach == 'minimal':
                prompt = chat_ocr._get_minimal_formatting_prompt()
                result = chat_ocr.process_pdf_with_chat(args.pdf_path, prompt, args.pages)
            elif args.approach == 'llamaparse':
                prompt = chat_ocr._get_llamaparse_formatting_prompt()
                result = chat_ocr.process_pdf_with_chat(args.pdf_path, prompt, args.pages)
            elif args.approach == 'italic_focused':
                prompt = chat_ocr._get_italic_focused_prompt()
                result = chat_ocr.process_pdf_with_chat(args.pdf_path, prompt, args.pages)
            elif args.approach == 'style_hunter':
                prompt = chat_ocr._get_style_hunter_prompt()
                result = chat_ocr.process_pdf_with_chat(args.pdf_path, prompt, args.pages)
            elif args.approach == 'ultra_precise':
                prompt = chat_ocr._get_ultra_precise_prompt()
                result = chat_ocr.process_pdf_with_chat(args.pdf_path, prompt, args.pages)
            elif args.approach == 'laser_focused':
                prompt = chat_ocr._get_laser_focused_prompt()
                result = chat_ocr.process_pdf_with_chat(args.pdf_path, prompt, args.pages)
            
            # Save results
            output_file = output_dir / f"{pdf_name}_{args.approach}_{timestamp}.md"
            metadata_file = output_dir / f"{pdf_name}_{args.approach}_{timestamp}_metadata.json"
            
            if result['status'] == 'success':
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result['markdown_content'])
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)
            
            console.print(f"[green]Results saved to: {output_file}[/green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)



if __name__ == "__main__":
    main()