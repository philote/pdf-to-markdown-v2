# Job Tracking Directory

This directory contains JSON files that track PDF to Markdown conversion jobs.

## Structure

Each job is stored as a JSON file with the naming pattern:
```
{job_id}.json
```

## Job Data Format

```json
{
  "id": "job_20240713_103045_rulebook",
  "input_file": "/path/to/input.pdf",
  "status": "completed|failed|started",
  "start_time": "2024-07-13T10:30:45.123456",
  "end_time": "2024-07-13T10:32:15.654321",
  "page_ranges": "1-10,15-20",
  "analysis": {
    "file_size_mb": 25.6,
    "total_pages": 100,
    "total_images": 45
  },
  "optimization": {
    "status": "success",
    "size_reduction_percent": 45.2,
    "images_removed": 45
  },
  "ocr_result": {
    "status": "success",
    "processing_time": 42.3,
    "model": "mistral-ocr-latest"
  },
  "output_files": {
    "markdown": "/path/to/output.md",
    "metadata": "/path/to/output_metadata.json"
  }
}
```

## Commands

List recent jobs:
```bash
python pdf2md.py --list-jobs
```

Check specific job:
```bash
python pdf2md.py --check-job job_20240713_103045_rulebook
```

## Cleanup

Old job files can be safely deleted to save space. The job tracking is for convenience and debugging only.