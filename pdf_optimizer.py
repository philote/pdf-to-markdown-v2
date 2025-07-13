#!/usr/bin/env python3
"""
PDF Optimizer for LLM Processing

Optimizes PDFs to reduce file size and processing time by:
- Removing/compressing images
- Removing unnecessary metadata
- Optimizing for text extraction
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import fitz  # PyMuPDF
from PyPDF2 import PdfReader, PdfWriter


class PDFOptimizer:
    """PDF optimizer focused on text extraction and size reduction"""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize PDF optimizer
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
    
    def optimize_for_text_extraction(self, input_path: str, output_path: Optional[str] = None, 
                                   remove_images: bool = True, compress_level: int = 3) -> Dict[str, Any]:
        """
        Optimize PDF for text extraction by removing/compressing images
        
        Args:
            input_path: Path to input PDF
            output_path: Path for optimized PDF (optional, creates temp file if None)
            remove_images: Remove all images (recommended for pure text extraction)
            compress_level: Compression level 1-9 (higher = more compression)
            
        Returns:
            Dict with optimization results and metadata
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input PDF not found: {input_path}")
        
        # Generate output path if not provided
        if output_path is None:
            output_dir = input_path.parent / "optimized"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{input_path.stem}_optimized.pdf"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        original_size = input_path.stat().st_size
        
        if self.verbose:
            print(f"Optimizing PDF for faster processing")
            print(f"Input: {input_path.name} ({original_size / 1024 / 1024:.1f} MB)")
            print(f"Output: {output_path}")
        
        try:
            # Use PyMuPDF for image removal and optimization
            if remove_images:
                result = self._remove_images_and_optimize(input_path, output_path, compress_level)
            else:
                result = self._compress_only(input_path, output_path, compress_level)
            
            optimized_size = output_path.stat().st_size
            size_reduction = ((original_size - optimized_size) / original_size) * 100
            
            result.update({
                'status': 'success',
                'input_file': str(input_path),
                'output_file': str(output_path),
                'original_size_mb': original_size / 1024 / 1024,
                'optimized_size_mb': optimized_size / 1024 / 1024,
                'size_reduction_percent': size_reduction,
                'optimization_type': 'remove_images' if remove_images else 'compress_only'
            })
            
            if self.verbose:
                print(f"Optimization complete")
                print(f"Size reduction: {size_reduction:.1f}% ({original_size / 1024 / 1024:.1f} MB → {optimized_size / 1024 / 1024:.1f} MB)")
                if result.get('images_removed', 0) > 0:
                    print(f"Removed {result['images_removed']} images")
            
            return result
            
        except Exception as e:
            error_result = {
                'status': 'error',
                'input_file': str(input_path),
                'error': str(e),
                'error_type': type(e).__name__
            }
            
            if self.verbose:
                print(f"Optimization failed: {e}")
            
            return error_result
    
    def _remove_images_and_optimize(self, input_path: Path, output_path: Path, compress_level: int) -> Dict[str, Any]:
        """Remove all images and optimize PDF using PyMuPDF"""
        doc = fitz.open(str(input_path))
        images_removed = 0
        pages_processed = 0
        
        try:
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Get all images on the page
                image_list = page.get_images()
                
                # Remove each image
                for img_index, img in enumerate(image_list):
                    try:
                        # Remove image by replacing with empty content
                        xref = img[0]  # xref number of the image
                        page.delete_image(xref)
                        images_removed += 1
                    except Exception as e:
                        if self.verbose:
                            print(f"Could not remove image {img_index} on page {page_num + 1}: {e}")
                
                pages_processed += 1
            
            # Save with compression
            doc.save(str(output_path), 
                    garbage=4,  # Remove unused objects
                    deflate=True,  # Compress content streams
                    clean=True,  # Clean up document
                    ascii=False)  # Keep Unicode
            
            return {
                'images_removed': images_removed,
                'pages_processed': pages_processed,
                'compression_level': compress_level
            }
            
        finally:
            doc.close()
    
    def _compress_only(self, input_path: Path, output_path: Path, compress_level: int) -> Dict[str, Any]:
        """Compress PDF without removing images"""
        doc = fitz.open(str(input_path))
        
        try:
            # Save with compression but keep images
            doc.save(str(output_path),
                    garbage=4,  # Remove unused objects
                    deflate=True,  # Compress content streams  
                    clean=True,  # Clean up document
                    ascii=False)  # Keep Unicode
            
            return {
                'images_removed': 0,
                'pages_processed': doc.page_count,
                'compression_level': compress_level
            }
            
        finally:
            doc.close()
    
    def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze PDF to understand size, images, and optimization potential
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dict with PDF analysis results
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        doc = fitz.open(str(pdf_path))
        
        try:
            total_images = 0
            page_info = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                images_on_page = len(page.get_images())
                text_length = len(page.get_text())
                
                total_images += images_on_page
                page_info.append({
                    'page': page_num + 1,
                    'images': images_on_page,
                    'text_length': text_length,
                    'exceeds_64kb': text_length > 64000  # LlamaParse limit
                })
            
            file_size_mb = pdf_path.stat().st_size / 1024 / 1024
            
            analysis = {
                'file_path': str(pdf_path),
                'file_size_mb': file_size_mb,
                'total_pages': doc.page_count,
                'total_images': total_images,
                'avg_images_per_page': total_images / doc.page_count if doc.page_count > 0 else 0,
                'pages_with_images': sum(1 for p in page_info if p['images'] > 0),
                'pages_exceeding_text_limit': sum(1 for p in page_info if p['exceeds_64kb']),
                'exceeds_llamaparse_limits': {
                    'file_size': file_size_mb > 300,
                    'has_image_heavy_pages': any(p['images'] > 35 for p in page_info),
                    'has_text_heavy_pages': any(p['exceeds_64kb'] for p in page_info)
                },
                'optimization_recommendations': [],
                'page_details': page_info
            }
            
            # Generate recommendations
            if analysis['total_images'] > 0:
                analysis['optimization_recommendations'].append("Remove images to reduce file size and processing time")
            
            if file_size_mb > 200:  # Approaching 300MB limit
                analysis['optimization_recommendations'].append("File size approaching LlamaParse limit - optimization recommended")
            
            if analysis['pages_exceeding_text_limit'] > 0:
                analysis['optimization_recommendations'].append(f"{analysis['pages_exceeding_text_limit']} pages may exceed 64KB text limit")
            
            if self.verbose:
                print(f"PDF Analysis: {pdf_path.name}")
                print(f"  Pages: {analysis['total_pages']}")
                print(f"  Size: {file_size_mb:.1f} MB")
                print(f"  Images: {analysis['total_images']} total, {analysis['avg_images_per_page']:.1f} avg/page")
                if analysis['optimization_recommendations']:
                    print(f"  Recommendations:")
                    for rec in analysis['optimization_recommendations']:
                        print(f"    • {rec}")
            
            return analysis
            
        finally:
            doc.close()
    
    def should_optimize(self, pdf_path: str, size_threshold_mb: float = 50) -> bool:
        """
        Determine if PDF should be optimized based on size and image content
        
        Args:
            pdf_path: Path to PDF file
            size_threshold_mb: Size threshold in MB above which optimization is recommended
            
        Returns:
            True if optimization is recommended
        """
        try:
            analysis = self.analyze_pdf(pdf_path)
            
            # Recommend optimization if:
            # - File is large
            # - Has many images
            # - Approaching LlamaParse limits
            should_opt = (
                analysis['file_size_mb'] > size_threshold_mb or
                analysis['total_images'] > 10 or
                any(analysis['exceeds_llamaparse_limits'].values())
            )
            
            return should_opt
            
        except Exception as e:
            if self.verbose:
                print(f"Could not analyze PDF for optimization decision: {e}")
            return False


def main():
    """Test the optimizer with a simple example"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_optimizer.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    optimizer = PDFOptimizer(verbose=True)
    
    try:
        # Analyze PDF
        print("=" * 50)
        analysis = optimizer.analyze_pdf(pdf_path)
        
        # Recommend optimization
        if optimizer.should_optimize(pdf_path):
            print("\n🎯 Optimization recommended")
            result = optimizer.optimize_for_text_extraction(pdf_path)
            if result['status'] == 'success':
                print(f"Optimized PDF saved to: {result['output_file']}")
        else:
            print("\nPDF already optimized for text extraction")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()