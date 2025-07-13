# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based tool for converting PDF TTRPG (tabletop role-playing game) documents to clean, formatted Markdown using the Mistral OCR API. The project focuses on aggressive formatting preservation and automatic PDF optimization.

## Key Commands

### Main PDF to Markdown Conversion

#### Standard OCR Approach (Original)
```bash
python pdf2md.py <pdf_path>
```

#### Advanced Chat API Approach (Recommended for Better Formatting)
```bash
python pdf2md_chat.py <pdf_path>
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Testing Individual Components
```bash
python pdf_optimizer.py <pdf_path>    # Test PDF optimization
python mistral_ocr.py <pdf_path>      # Test OCR module (standard)
```

### Advanced Chat API Options
```bash
# Different formatting approaches
python pdf2md_chat.py <pdf_path> --approach laser_focused    # Best for bold/italic detection
python pdf2md_chat.py <pdf_path> --approach italic_focused   # Enhanced italic detection
python pdf2md_chat.py <pdf_path> --approach llamaparse       # LlamaParse-style prompts

# Page range processing (cost optimization)
python pdf2md_chat.py <pdf_path> --pages "5-8"
python pdf2md_chat.py <pdf_path> --pages "1-10,15-20"

# Custom prompts
python pdf2md_chat.py <pdf_path> --custom-prompt my_prompt.txt

# Test all approaches for comparison
python pdf2md_chat.py <pdf_path> --test-all --pages "5-8"
```

### Job Management
```bash
# Standard approach
python pdf2md.py --list-jobs          # List recent jobs
python pdf2md.py --check-job <id>     # Check job status

# Chat API approach  
python pdf2md_chat.py --list-jobs     # List recent jobs
python pdf2md_chat.py --check-job <id># Check job status
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

#### 3. **pdf2md.py**: Original CLI Application
- `PDFToMarkdownConverter` class: Orchestrates the complete pipeline with standard OCR
- Features:
  - Rich CLI interface with progress bars
  - Job tracking and history
  - Error handling and recovery
  - Metadata generation

#### 4. **pdf2md_chat.py**: Advanced Chat API Application
- `PDFToMarkdownChatConverter` class: Superior formatting preservation using Mistral Chat API
- Features:
  - Multiple formatting approaches (8 different strategies)
  - Superior italic and bold-italic detection
  - Custom prompt support
  - Advanced job tracking with approach metadata
  - Test all approaches functionality
  - Page range optimization for cost savings

### Directory Structure
- `input/`: Place PDF files here for processing (test file: `AW-Basic_Refbook.pdf`)
- `output/`: Processed Markdown files and metadata JSON
- `optimized/`: Auto-created directory for optimized PDFs
- `.jobs/`: Job tracking files for history and debugging

### Processing Pipeline

#### Standard Pipeline (pdf2md.py)
1. **Input Validation**: Check PDF exists and parse options
2. **PDF Analysis**: Analyze structure, size, images using PDFOptimizer
3. **Optimization**: Remove images and compress if beneficial
4. **OCR Processing**: Convert to Markdown via Mistral OCR API
5. **Output Generation**: Save Markdown file and processing metadata
6. **Job Tracking**: Record job details for future reference

#### Advanced Chat Pipeline (pdf2md_chat.py)
1. **Input Validation**: Check PDF exists and parse options
2. **Job Initialization**: Create job ID and tracking metadata
3. **Page Extraction**: Extract specific pages if range specified (cost optimization)
4. **Prompt Selection**: Choose formatting approach or use custom prompt
5. **File Upload**: Upload PDF to Mistral API with signed URL
6. **Chat Processing**: Process with document + custom formatting prompts
7. **Advanced Output**: Save with job tracking and comprehensive metadata
8. **Results Display**: Show formatting statistics and job completion

## Important Context

### Formatting Preservation

#### Standard OCR Approach
Limited formatting preservation due to API constraints. Basic bold detection but struggles with italic and bold-italic combinations.

#### Advanced Chat API Approach
Superior formatting preservation with multiple specialized approaches:

1. **laser_focused** (Recommended): Best overall detection for bold, italic, and bold-italic
   - Successfully detects: `***stabilize and heal someone at 9:00 or past***`
   - Successfully detects: `*a night in high luxury & company*`
   - 27 bold, 29 italic, 2 bold-italic elements detected

2. **italic_focused**: Enhanced italic detection with visual analysis
   - 61 bold elements detected (5x improvement over standard)

3. **llamaparse**: LlamaParse-proven formatting prompt
   - Proven effective: "MANDATORY FORMATTING REQUIREMENTS: You MUST NEVER ignore text formatting..."

4. **Other approaches**: aggressive, style_hunter, ultra_precise, default, minimal

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

#### Standard Approach
```bash
# Basic conversion
python pdf2md.py input/AW-Basic_Refbook.pdf --pages 1-5

# Check optimization decisions
python pdf_optimizer.py input/AW-Basic_Refbook.pdf

# Test OCR directly
python mistral_ocr.py input/AW-Basic_Refbook.pdf
```

#### Advanced Chat Approach (Recommended)
```bash
# Best formatting preservation
python pdf2md_chat.py input/AW-Basic_Refbook.pdf --approach laser_focused

# Test specific pages for cost optimization
python pdf2md_chat.py input/AW-Basic_Refbook.pdf --pages 5-8 --approach laser_focused

# Compare all approaches (for research/testing)
python pdf2md_chat.py input/AW-Basic_Refbook.pdf --test-all --pages 5-8

# Custom formatting prompts
python pdf2md_chat.py input/AW-Basic_Refbook.pdf --custom-prompt my_custom_prompt.txt

# Job management
python pdf2md_chat.py --list-jobs
python pdf2md_chat.py --check-job job_20250713_162124_AW-Basic_Refbook
```