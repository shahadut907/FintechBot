"""
Document Processing System for FinSolve RAG Chatbot
Handles markdown, CSV, and other document formats
"""
import os
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import re
from src.config.settings import settings
from src.config.roles import DOCUMENT_CATEGORIES, Department
import io
import warnings


class DocumentChunk:
    """Represents a chunk of processed document"""
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata
        
    def __repr__(self):
        return f"DocumentChunk(length={len(self.content)}, source={self.metadata.get('source', 'unknown')})"


class DocumentProcessor:
    """Processes various document formats for RAG system"""
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        
    def process_markdown(self, file_path: str) -> List[DocumentChunk]:
        """Process markdown files into chunks"""
        print(f"📄 Processing markdown: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Get document metadata
            filename = os.path.basename(file_path)
            doc_meta = DOCUMENT_CATEGORIES.get(filename, {
                "department": Department.GENERAL,
                "type": "unknown"
            })
            
            # Split by headers first (better semantic chunking)
            chunks = self._smart_chunk_markdown(content)
            
            # Create DocumentChunk objects
            document_chunks = []
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:  # Only keep substantial chunks
                    metadata = {
                        "source": filename,
                        "chunk_id": i,
                        "department": doc_meta.get("department"),
                        "doc_type": doc_meta.get("type"),
                        "content_type": "markdown"
                    }
                    document_chunks.append(DocumentChunk(chunk, metadata))
            
            print(f"✅ Created {len(document_chunks)} chunks from {filename}")
            return document_chunks
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return []
    
    def process_csv(self, file_path: str) -> List[DocumentChunk]:
        """Process CSV files into searchable chunks"""
        print(f"📊 Processing CSV: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            filename = os.path.basename(file_path)
            
            # Get document metadata
            doc_meta = DOCUMENT_CATEGORIES.get(filename, {
                "department": Department.GENERAL,
                "type": "data"
            })
            
            chunks = []
            
            # Create summary chunk
            summary = self._create_csv_summary(df, filename)
            summary_metadata = {
                "source": filename,
                "chunk_id": 0,
                "department": doc_meta.get("department"),
                "doc_type": "data_summary",
                "content_type": "csv_summary"
            }
            chunks.append(DocumentChunk(summary, summary_metadata))
            
            # Create chunks for groups of rows (if dataset is large)
            if len(df) > 20:  # For large datasets, group rows
                chunk_size = 10  # Rows per chunk
                for i in range(0, len(df), chunk_size):
                    chunk_df = df.iloc[i:i+chunk_size]
                    chunk_content = self._format_csv_chunk(chunk_df, i, filename)
                    
                    metadata = {
                        "source": filename,
                        "chunk_id": i // chunk_size + 1,
                        "department": doc_meta.get("department"),
                        "doc_type": "data_rows",
                        "content_type": "csv_data",
                        "row_range": f"{i}-{min(i+chunk_size-1, len(df)-1)}"
                    }
                    chunks.append(DocumentChunk(chunk_content, metadata))
            else:
                # For small datasets, include all data in one chunk
                all_data = self._format_csv_chunk(df, 0, filename)
                metadata = {
                    "source": filename,
                    "chunk_id": 1,
                    "department": doc_meta.get("department"),
                    "doc_type": "data_complete",
                    "content_type": "csv_data"
                }
                chunks.append(DocumentChunk(all_data, metadata))
            
            print(f"✅ Created {len(chunks)} chunks from {filename}")
            return chunks
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return []
    
    def process_pdf(self, file_path: str) -> List[DocumentChunk]:
        """Process PDF files into chunks, with OCR fallback for scanned pages"""
        print(f"📑 Processing PDF: {file_path}")

        filename = os.path.basename(file_path)
        doc_meta = DOCUMENT_CATEGORIES.get(filename, {
            "department": Department.GENERAL,
            "type": "report"
        })

        try:
            raw_text = self._extract_pdf_text(file_path)
            if not raw_text.strip():
                print(f"⚠️ No extractable text found in {filename}")
                return []

            cleaned_text = self._clean_text(raw_text)
            chunks = self._chunk_text(cleaned_text)

            document_chunks = []
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 10:
                    continue  # Skip trivial chunks
                metadata = {
                    "source": filename,
                    "chunk_id": i,
                    "department": doc_meta.get("department"),
                    "doc_type": doc_meta.get("type"),
                    "content_type": "pdf"
                }
                document_chunks.append(DocumentChunk(chunk, metadata))

            print(f"✅ Created {len(document_chunks)} chunks from {filename}")
            return document_chunks

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return []

    # ----------------- PDF Helper Methods ----------------- #
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF using PyMuPDF or pdfminer; OCR fallback for scanned pages."""
        text_pages: List[str] = []

        # Attempt PyMuPDF first (fast & accurate)
        try:
            import fitz  # type: ignore
            with fitz.open(file_path) as doc:
                for page in doc:
                    page_text = page.get_text("text")
                    if page_text:
                        text_pages.append(page_text)
            if text_pages:
                return "\n".join(text_pages)
        except Exception:
            pass  # Gracefully fall back

        # Attempt pdfminer.six as a secondary option
        try:
            from pdfminer.high_level import extract_pages  # type: ignore
            from pdfminer.layout import LTTextContainer  # type: ignore

            for page_layout in extract_pages(file_path):
                page_txt = ""
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        page_txt += element.get_text()
                if page_txt:
                    text_pages.append(page_txt)
            if text_pages:
                return "\n".join(text_pages)
        except Exception:
            pass  # Gracefully fall back

        # Attempt pdfplumber as another fallback (good for tricky PDFs)
        try:
            import pdfplumber  # type: ignore
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if page_text:
                        text_pages.append(page_text)
            if text_pages:
                return "\n".join(text_pages)
        except Exception:
            pass

        # Attempt PyPDF2 as a lightweight option
        try:
            from PyPDF2 import PdfReader  # type: ignore
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    text_pages.append(page_text)
            if text_pages:
                return "\n".join(text_pages)
        except Exception:
            pass

        # OCR fallback for scanned PDFs
        try:
            from pdf2image import convert_from_path  # type: ignore
            import pytesseract  # type: ignore

            images = convert_from_path(file_path, dpi=300)
            for img in images:
                page_text = pytesseract.image_to_string(img)
                if page_text:
                    text_pages.append(page_text)
            return "\n".join(text_pages)
        except Exception as e:
            print(f"⚠️ OCR fallback failed for {file_path}: {e}")

        # If all methods fail, return empty string
        return ""

    def _clean_text(self, text: str) -> str:
        """Clean extracted text by removing non-printable characters and normalising whitespace."""
        # Remove non-printable characters
        cleaned = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u00A0-\uD7FF\uE000-\uFFFD]', '', text)
        # Consolidate whitespace
        cleaned = re.sub(r'\s+\n', '\n', cleaned)
        cleaned = re.sub(r'\n\s+', '\n', cleaned)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        return cleaned.strip()

    def _chunk_text(self, text: str) -> List[str]:
        """Chunk plain text into manageable pieces following the configured chunk size."""
        paragraphs = text.split('\n\n')
        chunks: List[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) > self.chunk_size and current:
                chunks.append(current.strip())
                current = para
            else:
                current += ("\n\n" + para) if current else para

        if current.strip():
            chunks.append(current.strip())

        # Further split overly large chunks using existing helper
        final_chunks: List[str] = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                final_chunks.extend(self._split_large_chunk(chunk))

        return final_chunks
    
    def _smart_chunk_markdown(self, content: str) -> List[str]:
        """Smart chunking that respects markdown structure"""
        # Split by headers first
        sections = re.split(r'\n(?=#{1,6}\s)', content)
        
        chunks = []
        current_chunk = ""
        
        for section in sections:
            # If adding this section would exceed chunk size
            if len(current_chunk) + len(section) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                current_chunk += "\n" + section if current_chunk else section
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If any chunk is still too large, split it further
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                # Split large chunks by paragraphs or sentences
                sub_chunks = self._split_large_chunk(chunk)
                final_chunks.extend(sub_chunks)
        
        return final_chunks
    
    def _split_large_chunk(self, text: str) -> List[str]:
        """Split large chunks by paragraphs or sentences"""
        # Try splitting by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current = ""
        
        for para in paragraphs:
            if len(current) + len(para) > self.chunk_size and current:
                chunks.append(current.strip())
                current = para
            else:
                current += "\n\n" + para if current else para
        
        if current.strip():
            chunks.append(current.strip())
        
        return chunks
    
    def _create_csv_summary(self, df: pd.DataFrame, filename: str) -> str:
        """Create a summary of CSV data for better searchability"""
        summary_parts = [
            f"Dataset: {filename}",
            f"Total Records: {len(df)}",
            f"Columns: {', '.join(df.columns.tolist())}",
            ""
        ]
        
        # Add column statistics
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                summary_parts.append(f"{col}: min={df[col].min()}, max={df[col].max()}, avg={df[col].mean():.2f}")
            elif df[col].dtype == 'object':
                unique_count = df[col].nunique()
                summary_parts.append(f"{col}: {unique_count} unique values")
                if unique_count <= 10:
                    unique_vals = df[col].value_counts().head(5)
                    summary_parts.append(f"  Top values: {dict(unique_vals)}")
        
        return "\n".join(summary_parts)
    
    def _format_csv_chunk(self, df: pd.DataFrame, start_idx: int, filename: str) -> str:
        """Format CSV chunk for better readability"""
        lines = [f"Data from {filename} (rows {start_idx} to {start_idx + len(df) - 1}):"]
        lines.append("")
        
        for idx, row in df.iterrows():
            row_lines = [f"Record {idx}:"]
            for col, val in row.items():
                row_lines.append(f"  {col}: {val}")
            lines.append("\n".join(row_lines))
            lines.append("")
        
        return "\n".join(lines)
    
    def process_directory(self, directory_path: str) -> List[DocumentChunk]:
        """Process all documents in a directory"""
        print(f"📁 Processing directory: {directory_path}")
        
        all_chunks = []
        directory = Path(directory_path)
        
        if not directory.exists():
            print(f"❌ Directory not found: {directory_path}")
            return []
        
        # Process markdown files
        for md_file in directory.glob("*.md"):
            chunks = self.process_markdown(str(md_file))
            all_chunks.extend(chunks)
        
        # Process CSV files
        for csv_file in directory.glob("*.csv"):
            chunks = self.process_csv(str(csv_file))
            all_chunks.extend(chunks)
        
        # Process PDF files
        for pdf_file in directory.glob("*.pdf"):
            chunks = self.process_pdf(str(pdf_file))
            all_chunks.extend(chunks)
        
        print(f"🎉 Total chunks created: {len(all_chunks)}")
        return all_chunks


# Test the processor
if __name__ == "__main__":
    print("🧪 Testing Document Processor")
    print("=" * 40)
    
    processor = DocumentProcessor()
    
    # Create sample test file
    test_dir = "src/data/raw"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a test markdown file
    test_content = """# Test Document
    
## Introduction
This is a test document for the FinSolve RAG system.

## Section 1
Here we discuss various company policies and procedures.

### Subsection 1.1
Detailed information about leave policies.

## Section 2
Financial information and budgets.
"""
    
    test_file = os.path.join(test_dir, "test_doc.md")
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    # Test processing
    chunks = processor.process_markdown(test_file)
    
    print(f"\n📊 Test Results:")
    print(f"Created {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  Length: {len(chunk.content)} chars")
        print(f"  Metadata: {chunk.metadata}")
        print(f"  Content preview: {chunk.content[:100]}...")
    
    # Clean up
    os.remove(test_file)
    print(f"\n✅ Document processor test completed!")