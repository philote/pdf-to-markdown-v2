# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based tool for converting PDF TTRPG (tabletop role-playing game) documents to clean, formatted Markdown using the Mistral OCR API. The project focuses on aggressive formatting preservation and automatic PDF optimization.

## Key Commands

### Main PDF to Markdown Conversion
```bash
python pdf2md.py <pdf_path>
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Testing Individual Components
```bash
python pdf_optimizer.py <pdf_path>    # Test PDF optimization
python mistral_ocr.py <pdf_path>      # Test OCR module
```

### Job Management
```bash
python pdf2md.py --list-jobs          # List recent jobs
python pdf2md.py --check-job <id>     # Check job status
```

## Architecture

### Complete Implementation
The project now consists of three main components working together:

#### 1. **pdf_optimizer.py**: PDF Analysis and Optimization
- `PDFOptimizer` class: Main interface for PDF processing
- Key methods:
  - `analyze_pdf()`: Analyzes PDF structure, size, and content
  - `optimize_for_text_extraction()`: Removes images and compresses PDFs
  - `should_optimize()`: Determines if optimization is needed

#### 2. **mistral_ocr.py**: Mistral OCR API Integration
- `MistralOCR` class: Handles PDF to Markdown conversion
- Key methods:
  - `process_pdf()`: Main conversion method with progress tracking
  - `process_with_formatting_prompt()`: Conversion with formatting preservation
  - `check_capabilities()`: Verify API status and features

#### 3. **pdf2md.py**: Main CLI Application
- `PDFToMarkdownConverter` class: Orchestrates the complete pipeline
- Features:
  - Rich CLI interface with progress bars
  - Job tracking and history
  - Error handling and recovery
  - Metadata generation

### Directory Structure
- `input/`: Place PDF files here for processing (test file: `AW-Basic_Refbook.pdf`)
- `output/`: Processed Markdown files and metadata JSON
- `optimized/`: Auto-created directory for optimized PDFs
- `.jobs/`: Job tracking files for history and debugging

### Processing Pipeline
1. **Input Validation**: Check PDF exists and parse options
2. **PDF Analysis**: Analyze structure, size, images using PDFOptimizer
3. **Optimization**: Remove images and compress if beneficial
4. **OCR Processing**: Convert to Markdown via Mistral OCR API
5. **Output Generation**: Save Markdown file and processing metadata
6. **Job Tracking**: Record job details for future reference

## Important Context

### Formatting Preservation
The README includes a critical prompt for formatting preservation when using LLM-based parsing:
```python
instruction = "MANDATORY FORMATTING REQUIREMENTS: You MUST NEVER ignore text formatting. ALWAYS convert bold text to **bold** markdown and italic text to *italic* markdown. FAILURE TO PRESERVE FORMATTING IS UNACCEPTABLE. Scan every single word for font weight changes, emphasis, and styling. Bold headings, author names, game titles, and emphasized terms are CRITICAL and must be preserved. This is a STRICT REQUIREMENT - do not skip any formatted text."
```

### PDF Processing Limits
The optimizer checks for LlamaParse-compatible limits:
- Maximum file size: 300MB
- Maximum images per page: 35
- Maximum text per page: 64KB

### Environment Configuration
The project uses a `.env` file (likely for Mistral API credentials). Create this file with necessary API keys before implementing the Mistral OCR integration.

## Development Notes

### Current Status
- **Complete implementation** with all core features working
- PDF optimization, OCR integration, and CLI interface fully implemented
- Job tracking and progress display functional
- Comprehensive error handling and user feedback

### Dependencies
- All dependencies defined in `requirements.txt`
- Environment configuration via `.env` file
- API key required: `MISTRAL_API_KEY`

### Testing
- Test file available: `input/AW-Basic_Refbook.pdf`
- Individual component testing supported
- Integration testing via main CLI tool

### Usage Examples
```bash
# Test with sample file
python pdf2md.py input/AW-Basic_Refbook.pdf --pages 1-5

# Check optimization decisions
python pdf_optimizer.py input/AW-Basic_Refbook.pdf

# Test OCR directly
python mistral_ocr.py input/AW-Basic_Refbook.pdf
```