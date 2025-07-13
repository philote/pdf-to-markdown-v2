# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based tool for converting PDF TTRPG (tabletop role-playing game) documents to clean, formatted Markdown using the Mistral OCR API. The project focuses on aggressive formatting preservation and automatic PDF optimization.

## Key Commands

### Running the PDF Optimizer
```bash
python pdf_optimizer.py <pdf_path>
```

### Installing Dependencies
```bash
pip install PyMuPDF PyPDF2
```

Note: The project currently lacks a requirements.txt file. When implementing the Mistral OCR integration, additional dependencies will be needed.

## Architecture

### Current Implementation
- **pdf_optimizer.py**: Core module providing PDF analysis and optimization functionality
  - `PDFOptimizer` class: Main interface for PDF processing
  - Key methods:
    - `analyze_pdf()`: Analyzes PDF structure, size, and content
    - `optimize_for_text_extraction()`: Removes images and compresses PDFs
    - `should_optimize()`: Determines if optimization is needed

### Directory Structure
- `input/`: Place PDF files here for processing
- `output/`: Processed Markdown files will be saved here
- `optimized/`: Auto-created directory for optimized PDFs

### Planned Architecture
The complete pipeline will involve:
1. PDF analysis and optimization (implemented)
2. Mistral OCR API integration for PDF to Markdown conversion (not yet implemented)
3. Post-processing for formatting preservation and cleanup

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

- The project is in early development with only the PDF optimization component implemented
- No testing framework is currently set up
- No dependency management system (requirements.txt, Pipfile, etc.) exists yet
- The main PDF to Markdown conversion functionality using Mistral OCR needs to be implemented