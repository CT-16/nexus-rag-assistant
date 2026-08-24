import os
import tempfile
from typing import List

# --- UPDATED 2026 IMPORTS ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentManager:
    """Handles PDF uploading, text extraction, and semantic chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_uploaded_files(self, uploaded_files) -> List[Document]:
        """Saves uploaded stream files to temporary storage, parses pages, and splits into chunks."""
        all_chunks = []
        
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                
                for doc in documents:
                    doc.metadata["source_file"] = uploaded_file.name
                    
                chunks = self.text_splitter.split_documents(documents)
                all_chunks.extend(chunks)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        return all_chunks