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

Process specific pages:
```bash
python pdf2md.py input/document.pdf --pages 1-10,15-20
```

Skip optimization (for already optimized PDFs):
```bash
python pdf2md.py input/document.pdf --no-optimize
```

## CLI Reference

### Main Commands

```bash
# Convert PDF to Markdown
python pdf2md.py <pdf_file> [options]

# List recent processing jobs
python pdf2md.py --list-jobs

# Check specific job status
python pdf2md.py --check-job <job_id>
```

### Options

- `--pages`: Specify page ranges (e.g., "1-10,15-20")
- `--no-optimize`: Skip PDF optimization step
- `--keep-images`: Keep images during optimization
- `--verbose`: Enable detailed output

### Examples

```bash
# Basic conversion
python pdf2md.py input/rulebook.pdf

# Convert specific pages only
python pdf2md.py input/rulebook.pdf --pages 1-50,75-100

# Skip optimization for small files
python pdf2md.py input/small_doc.pdf --no-optimize

# Keep images (useful for illustrated content)
python pdf2md.py input/bestiary.pdf --keep-images

# Check what jobs have been run
python pdf2md.py --list-jobs

# Get details about a specific job
python pdf2md.py --check-job job_20240713_103045_rulebook
```

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

### Mistral OCR
- **Documentation**: https://docs.mistral.ai/capabilities/OCR/basic_ocr/
- **Pricing**: ~1000 pages per dollar
- **Model**: mistral-ocr-latest
- **Features**: Multilingual, formatting preservation, table extraction

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
```bash
# Test with the included sample
python pdf2md.py input/AW-Basic_Refbook.pdf --pages 1-5

# Test optimization
python pdf_optimizer.py input/AW-Basic_Refbook.pdf

# Test OCR module
python mistral_ocr.py input/AW-Basic_Refbook.pdf
```

## References

- [Mistral OCR Announcement](https://mistral.ai/news/mistral-ocr)
- [Mistral OCR Documentation](https://docs.mistral.ai/capabilities/OCR/basic_ocr/)
- [Notebook Examples](https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/ocr/structured_ocr.ipynb)

## License

MIT License - see LICENSE file for details.