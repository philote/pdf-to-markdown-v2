# PDF to Markdown Processor

This project is meant to create a tool that converts PDF TTRPG documents to clean formatted Markdown using Mistral OCR API with automatic optimization and styling detection.

## Features

**Style Detection**: Aggressive formatting preservation (bold, italic, headers, lists, and tables)
**Clean Output**: Remove page numbers, dividers, and optimize bullet points
**PDF Optimization**: Automatic image removal and compression for faster processing
**Page Range Processing**: Process specific page ranges to save credits
**Job Tracking**: Track Mistral OCR job IDs for debugging and analytics
**Detailed Metadata**: Comprehensive processing results and statistics

Mistral OCR
https://mistral.ai/news/mistral-ocr

Mistral OCR documentation
https://docs.mistral.ai/capabilities/OCR/basic_ocr/

Notebook examples:
https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/ocr/structured_ocr.ipynb#scrollTo=NhwM0aITt7ti


## Notes:
In a similar project that uses the LlamaParse API we found this prompt helpful:
``` python
instruction = "MANDATORY FORMATTING REQUIREMENTS: You MUST NEVER ignore text formatting. ALWAYS convert bold text to **bold** markdown and italic text to *italic* markdown. FAILURE TO PRESERVE FORMATTING IS UNACCEPTABLE. Scan every single word for font weight changes, emphasis, and styling. Bold headings, author names, game titles, and emphasized terms are CRITICAL and must be preserved. This is a STRICT REQUIREMENT - do not skip any formatted text."
```