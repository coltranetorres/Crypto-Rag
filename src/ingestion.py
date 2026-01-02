"""
Document ingestion pipeline with PDF support and ClusterSemanticChunker
Works with OpenRouter embeddings
"""
from typing import List, Dict, Any, Optional
import hashlib
from tqdm import tqdm
import time
import os
from pypdf import PdfReader
from chromadb import EmbeddingFunction, Documents, Embeddings
from src.config.logging_config import get_logger

logger = get_logger(__name__)

# Try to import ClusterSemanticChunker
try:
    from chunking_evaluation.chunking import ClusterSemanticChunker
    CLUSTER_CHUNKER_AVAILABLE = True
except ImportError:
    CLUSTER_CHUNKER_AVAILABLE = False
    logger.warning("chunking_evaluation not available. Install with: pip install git+https://github.com/brandonstarxel/chunking_evaluation.git")


class PDFDocumentLoader:
    """Load PDF documents from directory"""
    
    def __init__(self, directory: str = ".data"):
        self.directory = directory
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load PDF documents from directory"""
        documents = []
        
        if not os.path.exists(self.directory):
            logger.warning(f"Directory {self.directory} does not exist")
            return documents
        
        pdf_files = []
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.directory}")
            return documents
        
        logger.info(f"Found {len(pdf_files)} PDF file(s)")
        
        for filepath in tqdm(pdf_files, desc="Loading PDFs"):
            try:
                reader = PdfReader(filepath)
                text_content = []
                
                for page_num, page in enumerate(reader.pages, start=1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append({
                                "page": page_num,
                                "text": page_text
                            })
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num} of {filepath}: {e}")
                        continue
                
                if text_content:
                    # Combine all pages into one document with page markers
                    full_text = "\n\n".join([f"Page {p['page']}:\n{p['text']}" for p in text_content])
                    
                    # Store page information for later chunk tracking
                    page_info = {p['page']: p['text'] for p in text_content}
                    
                    documents.append({
                        "content": full_text,
                        "source": os.path.basename(filepath),
                        "filepath": filepath,
                        "total_pages": len(reader.pages),
                        "pages_with_text": len(text_content),
                        "page_content": text_content  # Store page-by-page content for tracking
                    })
                    
            except Exception as e:
                logger.error(f"Error loading PDF {filepath}: {e}", exc_info=True)
                continue
        
        logger.info(f"Loaded {len(documents)} PDF document(s)")
        return documents


class OpenRouterEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB EmbeddingFunction wrapper for OpenRouter embeddings
    Compatible with ClusterSemanticChunker
    """
    
    def __init__(self, embedding_wrapper):
        """
        Args:
            embedding_wrapper: OpenRouterEmbeddings instance
        """
        self.embedding_wrapper = embedding_wrapper
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for documents
        
        Args:
            input: List of text documents
            
        Returns:
            List of embedding vectors
        """
        if not input:
            return []
        
        # Use the embedding wrapper to get embeddings
        embeddings = self.embedding_wrapper.embed_documents(list(input))
        return embeddings


class DocumentProcessor:
    """Document processor with semantic chunking support"""
    
    def __init__(
        self,
        embedding_function: Optional[OpenRouterEmbeddingFunction] = None,
        use_semantic_chunking: bool = True,
        max_chunk_size: int = 400,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize document processor
        
        Args:
            embedding_function: OpenRouterEmbeddingFunction for semantic chunking
            use_semantic_chunking: Whether to use ClusterSemanticChunker (requires embedding_function)
            max_chunk_size: Max chunk size for ClusterSemanticChunker
            chunk_size: Chunk size for simple chunking (fallback)
            chunk_overlap: Chunk overlap for simple chunking (fallback)
        """
        self.embedding_function = embedding_function
        self.use_semantic_chunking = use_semantic_chunking and CLUSTER_CHUNKER_AVAILABLE
        self.max_chunk_size = max_chunk_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize semantic chunker if available
        if self.use_semantic_chunking and embedding_function:
            try:
                self.semantic_chunker = ClusterSemanticChunker(
                    embedding_function,
                    max_chunk_size=max_chunk_size
                )
                logger.info(f"ClusterSemanticChunker initialized (max_chunk_size={max_chunk_size})")
            except Exception as e:
                logger.warning(f"Failed to initialize ClusterSemanticChunker: {e}")
                logger.info("Falling back to simple chunking")
                self.use_semantic_chunking = False
        else:
            if self.use_semantic_chunking and not embedding_function:
                logger.warning("Semantic chunking requested but no embedding_function provided")
                self.use_semantic_chunking = False
            if self.use_semantic_chunking and not CLUSTER_CHUNKER_AVAILABLE:
                logger.warning("ClusterSemanticChunker not available, using simple chunking")
                self.use_semantic_chunking = False
    
    def load_documents(self, directory: str = ".data") -> List[Dict[str, Any]]:
        """Load PDF documents from directory"""
        loader = PDFDocumentLoader(directory)
        return loader.load_documents()
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text using semantic or simple chunking
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if self.use_semantic_chunking:
            try:
                # Use ClusterSemanticChunker
                chunks = self.semantic_chunker.split_text(text)
                return chunks
            except Exception as e:
                logger.warning(f"Semantic chunking failed: {e}, falling back to simple chunking")
                return self._simple_chunk_text(text)
        else:
            return self._simple_chunk_text(text)
    
    def _simple_chunk_text(self, text: str) -> List[str]:
        """Simple chunking with overlap (fallback method)"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk)
            
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def _extract_page_numbers(self, chunk_text: str, page_content: List[Dict]) -> List[int]:
        """
        Extract page numbers from chunk text by finding which pages it contains
        
        Args:
            chunk_text: The chunk text (may contain "Page X:\n" markers)
            page_content: List of dicts with 'page' and 'text' keys
            
        Returns:
            List of page numbers this chunk came from
        """
        pages = []
        
        # Check if chunk contains page markers
        if "Page " in chunk_text:
            # Extract page numbers from text
            import re
            page_matches = re.findall(r'Page (\d+):', chunk_text)
            if page_matches:
                pages = [int(p) for p in page_matches]
        else:
            # If no markers, try to infer from content overlap
            # Find which page texts overlap with this chunk
            chunk_lower = chunk_text.lower()
            for page_info in page_content:
                page_text = page_info['text'].lower()
                # Check if chunk contains significant portion of page text
                if len(page_text) > 0:
                    overlap_ratio = len(set(chunk_lower.split()) & set(page_text.split())) / max(len(set(chunk_lower.split())), 1)
                    if overlap_ratio > 0.3:  # 30% word overlap threshold
                        pages.append(page_info['page'])
        
        # Remove duplicates and sort
        return sorted(list(set(pages))) if pages else []
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """Process documents into chunks"""
        all_chunks = []
        
        for doc in tqdm(documents, desc="Processing documents"):
            chunks = self.chunk_text(doc['content'])
            page_content = doc.get('page_content', [])
            
            for i, chunk_text in enumerate(chunks):
                chunk_id = hashlib.md5(
                    f"{doc['source']}_{i}".encode()
                ).hexdigest()
                
                # Extract page numbers for this chunk
                page_numbers = self._extract_page_numbers(chunk_text, page_content)
                
                # Format source with page numbers
                source = doc['source']
                if page_numbers:
                    if len(page_numbers) == 1:
                        source_with_pages = f"{source} (page {page_numbers[0]})"
                    else:
                        source_with_pages = f"{source} (pages {min(page_numbers)}-{max(page_numbers)})"
                else:
                    source_with_pages = source
                
                all_chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source": source_with_pages,  # Include page numbers in source
                        "source_file": doc['source'],  # Original filename
                        "filepath": doc.get('filepath', ''),
                        "page_numbers": ",".join(map(str, page_numbers)) if page_numbers else "",  # Comma-separated string
                        "page_count": len(page_numbers),  # Number of pages
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunking_method": "semantic" if self.use_semantic_chunking else "simple",
                        "total_pages": doc.get('total_pages', 0),
                        "pages_with_text": doc.get('pages_with_text', 0)
                    }
                })
        
        logger.info(f"Created {len(all_chunks)} chunks using {('semantic' if self.use_semantic_chunking else 'simple')} chunking")
        return all_chunks


class ChromaIngestion:
    """Ingest chunks into ChromaDB"""
    
    def __init__(self, collection, embedding_wrapper, batch_size: int = 50):
        self.collection = collection
        self.embedding_wrapper = embedding_wrapper
        self.batch_size = batch_size
    
    def ingest(self, chunks: List[Dict]):
        """Ingest chunks with embeddings"""
        logger.info(f"Ingesting {len(chunks)} chunks into ChromaDB...")
        
        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Upserting"):
            batch = chunks[i:i + self.batch_size]
            
            texts = [c["text"] for c in batch]
            ids = [c["id"] for c in batch]
            metadatas = [c["metadata"] for c in batch]
            
            # Get embeddings
            embeddings = self.embedding_wrapper.embed_documents(texts)
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            # Rate limit
            time.sleep(0.5)
        
        total_count = self.collection.count()
        logger.info(f"Ingested {len(chunks)} chunks. Total in collection: {total_count}")
        return total_count

