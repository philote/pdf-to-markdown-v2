# PDF to Markdown Processor

A comprehensive CLI tool that converts PDF TTRPG documents to clean, formatted Markdown using Mistral OCR API with automatic optimization and progress tracking. Designed specifically for converting tabletop role-playing game PDFs into semantic markdown suitable for later conversion to Foundry VTT content.

## Features

- **Aggressive Format Preservation**: Bold, italic, headers, lists, and tables
- **Intelligent PDF Optimization**: Automatic image removal and compression for faster processing
- **Page Range Processing**: Process specific page ranges to save API credits
- **Progress Tracking**: Real-time progress display with rich CLI interface
- **Job Management**: Track processing history and check job status
- **Comprehensive Metadata**: Detailed processing results and statistics
- **Error Recovery**: Clear error messages with suggested fixes

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pdf-to-markdown-v2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API key:
```bash
cp .env.example .env
# Edit .env and add your Mistral API key
```

### Basic Usage

Convert a PDF to Markdown:
```bash
python pdf2md.py input/document.pdf
```

**Advanced formatting preservation** (recommended for TTRPG documents):
```bash
python pdf2md_chat.py input/document.pdf --approach laser_focused
```

Process specific pages:
```bash
python pdf2md.py input/document.pdf --pages 1-10,15-20
```

Skip optimization (for already optimized PDFs):
```bash
python pdf2md.py input/document.pdf --no-optimize
```

## CLI Reference

### Main Tools

**Standard Tool** (`pdf2md.py`) - Full pipeline with optimization:
```bash
# Convert PDF to Markdown with optimization
python pdf2md.py <pdf_file> [options]

# List recent processing jobs
python pdf2md.py --list-jobs

# Check specific job status
python pdf2md.py --check-job <job_id>
```

**Advanced Chat API Tool** (`pdf2md_chat.py`) - Superior formatting preservation:
```bash
# Advanced formatting preservation
python pdf2md_chat.py <pdf_file> [options]

# Test specific formatting approach
python pdf2md_chat.py <pdf_file> --approach <approach_name>

# Test all formatting approaches
python pdf2md_chat.py <pdf_file> --approach all

# Use custom formatting prompt
python pdf2md_chat.py <pdf_file> --custom-prompt prompt.txt

# Job management (same as pdf2md.py)
python pdf2md_chat.py --list-jobs
python pdf2md_chat.py --check-job <job_id>
```

### Options

**Standard Tool** (`pdf2md.py`):
- `--pages`: Specify page ranges (e.g., "1-10,15-20")
- `--no-optimize`: Skip PDF optimization step
- `--keep-images`: Keep images during optimization
- `--enhanced-formatting`: Enable enhanced formatting preservation for TTRPG documents
- `--verbose`: Enable detailed output

**Chat API Tool** (`pdf2md_chat.py`):
- `--pages`: Specify page ranges (e.g., "1-3,5,7-9")
- `--approach`: Choose formatting approach (see Formatting Approaches section)
- `--custom-prompt`: Use custom formatting prompt from file
- `--output-dir`: Output directory (default: "output")
- `--model`: Mistral model to use (default: "mistral-small-latest")

### Examples

**Standard Tool Examples:**
```bash
# Basic conversion
python pdf2md.py input/rulebook.pdf

# Convert specific pages only
python pdf2md.py input/rulebook.pdf --pages 1-50,75-100

# Skip optimization for small files
python pdf2md.py input/small_doc.pdf --no-optimize

# Keep images (useful for illustrated content)
python pdf2md.py input/bestiary.pdf --keep-images

# Enhanced formatting for TTRPG documents
python pdf2md.py input/rulebook.pdf --enhanced-formatting

# Check what jobs have been run
python pdf2md.py --list-jobs

# Get details about a specific job
python pdf2md.py --check-job job_20240713_103045_rulebook
```

**Chat API Tool Examples** (Superior formatting preservation):
```bash
# Best formatting preservation for TTRPG documents
python pdf2md_chat.py input/rulebook.pdf --approach laser_focused

# Test multiple formatting approaches
python pdf2md_chat.py input/rulebook.pdf --approach all --pages 5-8

# Aggressive formatting detection
python pdf2md_chat.py input/character_sheet.pdf --approach style_hunter

# Process specific pages with italic focus
python pdf2md_chat.py input/bestiary.pdf --pages 12-25 --approach italic_focused

# Use custom formatting prompt
python pdf2md_chat.py input/custom_doc.pdf --custom-prompt my_prompt.txt

# Quick test with minimal formatting
python pdf2md_chat.py input/simple_doc.pdf --approach minimal --pages 1-3

# Compare different approaches on same content
python pdf2md_chat.py input/test_doc.pdf --approach all --pages 1-2
```

## Advanced Chat API Tool (`pdf2md_chat.py`)

The Chat API tool uses Mistral's chat interface with document processing for **superior formatting preservation**. This approach is specifically designed for complex documents like TTRPG rulebooks where maintaining formatting (bold, italic, bold+italic) is critical.

### Key Advantages

- **Superior Format Detection**: Uses advanced prompting techniques to detect subtle formatting
- **Multiple Approaches**: 8 different formatting strategies optimized for different content types
- **Chat API Integration**: Leverages Mistral's document processing capabilities
- **Page Extraction**: Built-in PDF page extraction for targeted processing
- **Comprehensive Job Tracking**: Full job history and error handling

### Formatting Approaches

The tool includes 8 specialized formatting approaches:

| Approach | Best For | Description |
|----------|----------|-------------|
| `default` | General documents | Balanced formatting preservation |
| `aggressive` | Complex formatting | Zero-tolerance formatting detection |
| `minimal` | Simple documents | Basic structure preservation |
| `llamaparse` | TTRPG documents | Proven formatting patterns |
| `italic_focused` | Italic-heavy content | Enhanced italic and bold+italic detection |
| `style_hunter` | Complex layouts | Multi-layer formatting analysis |
| `ultra_precise` | Known formatting issues | Targets specific formatting patterns |
| `laser_focused` | **Recommended** | Optimized for TTRPG documents |

### When to Use Each Tool

**Use `pdf2md_chat.py` when:**
- Working with TTRPG documents (character sheets, rulebooks, bestiaries)
- Formatting preservation is critical (bold, italic, bold+italic text)
- You need to test multiple formatting approaches
- Processing specific page ranges for experimentation
- Documents have complex formatting that needs preservation

**Use `pdf2md.py` when:**
- Processing large documents that benefit from optimization
- Need full pipeline with PDF compression and image removal
- Working with standard documents where basic formatting is sufficient
- Want automatic optimization decisions

### Chat API Features

#### Page Range Processing
```bash
# Extract and process specific pages
python pdf2md_chat.py input/book.pdf --pages "1-3,7,12-15"
```

#### Approach Testing
```bash
# Test single approach
python pdf2md_chat.py input/doc.pdf --approach laser_focused

# Test all approaches (great for comparison)
python pdf2md_chat.py input/doc.pdf --approach all --pages 1-2
```

#### Custom Prompts
```bash
# Create custom_prompt.txt with your formatting instructions
python pdf2md_chat.py input/doc.pdf --custom-prompt custom_prompt.txt
```

#### Rich Progress Display
The tool provides real-time progress with:
- Multi-step progress bars (Upload → Process → Finalize)
- Processing time estimates
- Professional result panels
- Comprehensive error handling

### Output Structure

**Filename Pattern**: `{pdf_name}_{approach}_{job_id}.md`
**Metadata**: `{pdf_name}_{approach}_{job_id}_metadata.json`

**Comprehensive Metadata** includes:
- Processing approach and parameters
- Job tracking information
- Formatting prompt details
- Processing performance metrics
- Complete error information (if applicable)

## Setup Guide

### API Key Configuration

1. Get your Mistral API key from [Mistral Console](https://console.mistral.ai/)
2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your API key:
   ```bash
   MISTRAL_API_KEY=your-mistral-api-key-here
   ```

### Directory Structure

The tool uses these directories:
- `input/`: Place your PDF files here (or use any path)
- `output/`: Converted markdown files and metadata
- `optimized/`: Temporary optimized PDFs (auto-created)
- `.jobs/`: Processing job history (auto-created)

### Test File

A test file is included for validation: `input/AW-Basic_Refbook.pdf`

## Architecture

### Components

1. **pdf_optimizer.py**: PDF analysis and optimization
2. **mistral_ocr.py**: Mistral OCR API integration  
3. **pdf2md.py**: Main CLI application with progress tracking
4. **pdf2md_chat.py**: Advanced Chat API tool with superior formatting preservation

### Processing Pipeline

1. **Analysis**: Examine PDF structure, size, and optimization potential
2. **Optimization**: Remove images and compress (if beneficial)
3. **OCR Processing**: Convert to markdown via Mistral OCR
4. **Output**: Save markdown file and processing metadata

## Output Format

### Markdown File
Clean, semantic markdown with preserved formatting:
- Headers (`# ## ###`)
- Bold (`**text**`) and italic (`*text*`)
- Lists and tables
- Proper paragraph structure

#### Enhanced Formatting for TTRPG Documents
When using `--enhanced-formatting`, additional processing ensures:
- Game stats (cool, hard, hot, sharp, weird) are properly emphasized
- Move descriptions follow consistent formatting patterns
- Roll instructions are clearly highlighted
- TTRPG-specific terminology is properly formatted

### Metadata JSON
Comprehensive processing information:
```json
{
  "source_pdf": "rulebook.pdf",
  "output_markdown": "rulebook.md",
  "processing_date": "2024-07-13T10:30:00Z",
  "pages_processed": "1-50,75-100",
  "optimization": {
    "performed": true,
    "size_reduction": "45.2%",
    "images_removed": 127
  },
  "processing_time": {
    "optimization": 2.5,
    "ocr": 45.3,
    "total": 47.8
  }
}
```

## Troubleshooting

### Common Issues

**"MISTRAL_API_KEY not found"**
- Ensure your `.env` file exists and contains your API key
- Check that the key is valid on [Mistral Console](https://console.mistral.ai/)

**"PDF not found"**
- Verify the file path is correct
- Ensure the file is a valid PDF

**OCR processing fails**
- Check your internet connection
- Verify your API key has sufficient credits
- Try with a smaller page range first

**Large files timeout**
- Use `--pages` to process sections
- Enable optimization with image removal (default)

### Performance Tips

- Use page ranges for large documents to save processing time and costs
- Let optimization run (default) for files over 50MB
- Remove images (default) unless they contain critical text information

## API Integration

### Mistral APIs Used

**OCR API** (used by `pdf2md.py`):
- **Documentation**: https://docs.mistral.ai/capabilities/OCR/basic_ocr/
- **Model**: mistral-ocr-latest
- **Features**: Multilingual, formatting preservation, table extraction

**Chat API** (used by `pdf2md_chat.py`):
- **Documentation**: https://docs.mistral.ai/capabilities/completion/
- **Model**: mistral-small-latest (configurable)
- **Features**: Document processing, custom prompting, superior formatting control
- **Pricing**: ~1000 pages per dollar (similar to OCR API)

### Format Preservation
The tool uses aggressive formatting preservation based on successful TTRPG conversion patterns. The OCR process is optimized for:
- Game rule formatting
- Statistical tables
- Character sheets
- Bestiary entries

## Development

### Requirements
- Python 3.8+
- Dependencies listed in `requirements.txt`
- Mistral API key

### Testing

**Standard Tool:**
```bash
# Test with the included sample
python pdf2md.py input/AW-Basic_Refbook.pdf --pages 1-5

# Test optimization
python pdf_optimizer.py input/AW-Basic_Refbook.pdf

# Test OCR module
python mistral_ocr.py input/AW-Basic_Refbook.pdf
```

**Chat API Tool (Recommended for TTRPG content):**
```bash
# Test best formatting approach
python pdf2md_chat.py input/AW-Basic_Refbook.pdf --approach laser_focused --pages 5-8

# Compare multiple approaches
python pdf2md_chat.py input/AW-Basic_Refbook.pdf --approach all --pages 5-6

# Test italic detection (pages 5-8 contain italic and bold+italic text)
python pdf2md_chat.py input/AW-Basic_Refbook.pdf --approach italic_focused --pages 5-8
```

## References

- [Mistral OCR Announcement](https://mistral.ai/news/mistral-ocr)
- [Mistral OCR Documentation](https://docs.mistral.ai/capabilities/OCR/basic_ocr/)
- [Notebook Examples](https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/ocr/structured_ocr.ipynb)

## License

MIT License - see LICENSE file for details.