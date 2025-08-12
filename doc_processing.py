import os
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import asyncio

async def load_file(file_path: str | Path) -> list:
    """
    Load a single PDF file.
    
    Args:
        file_path (str | Path): Path to the PDF file
        
    Returns:
        list: List of pages from the PDF
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    if not file_path.suffix.lower() == '.pdf':
        raise ValueError(f"File must be a PDF: {file_path}")
        
    print(f"Processing {file_path.name}...")
    try:
        loader = PyPDFLoader(str(file_path))
        pages = []
        async for page in loader.alazy_load():
            pages.append(page)
        print(f"Loaded {len(pages)} pages from {file_path.name}")
        return pages
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")
        return []

async def load_path(directory_path: str | Path) -> list:
    """
    Load all PDF files from the specified directory.
    
    Args:
        directory_path (str | Path): Path to the directory containing PDF files
        
    Returns:
        list: List of all pages from all PDFs
    """
    # Convert to Path object
    pdf_dir = Path(directory_path)
    
    # List to store all pages from all PDFs
    all_pages = []
    
    # Get all PDF files in the directory
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return all_pages
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process each PDF file
    for pdf_file in pdf_files:
        pages = await load_file(pdf_file)
        all_pages.extend(pages)
    
    return all_pages

async def main():
    # Test loading a directory of PDFs
    pdf_dir = "resources/legco"
#    all_pages = await load_path(pdf_dir)
#    print(f"\nTotal pages loaded from directory: {len(all_pages)}")

    # Test loading a single PDF file
    single_pdf = Path(pdf_dir) / "s620192323p1.pdf"
    pages = await load_file(single_pdf)
    print(f"\nPages loaded from single file: {len(pages)}")

if __name__ == "__main__":
    asyncio.run(main())