"""
This module handles the vectorization and retrieval of documents for an agentic RAG system.

Key components and why they are used:
- MarkdownHeaderTextSplitter: Used for predictable retrieval testing by splitting documents based on headers, preserving document structure.
- RecursiveCharacterTextSplitter: Employed for semantic chunking with controlled chunk sizes (500-900 characters) and overlaps (80-120) to maintain context.
- OllamaEmbeddings with 'mxbai-embed-large:latest': Chosen as the embedding engine for generating high-quality vector representations.
- Chroma vectorstore: Utilized for storing and retrieving vectors with cosine similarity for efficient similarity searches.
- ContextualCompressionRetriever with CrossEncoderReranker: Implements reranking to avoid duplicates and improve retrieval quality, using MMR (Maximal Marginal Relevance) with top k=5.
- Additional utilities like Path, re, and shutil for file handling, text sanitization, and database management.
"""

import hashlib
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import shutil
import re
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# Directory to store Chroma vector database
CHROMA_DIR = Path(r"/Users/pardeepwalia/Desktop/Data/Agentic_RAG/Data/vector_db")
COLLECTION = "kb_md"


# compute a stable hash for dataset
def compute_dataset_hash(file_paths: list[str]) -> str:
    """
    Generates a stable hash for a dataset based on:
    - absolute file path
    - file size
    Any change in docs => new hash => new vector DB
    """
    hasher = hashlib.sha256()

    for path in sorted(file_paths):
        p = Path(path)
        if not p.exists():
            continue

        stat = p.stat()
        # Create a fingerprint using file name and size
        fingerprint = f"{p.name}|{stat.st_size}"
        hasher.update(fingerprint.encode("utf-8"))

    # Short hash keeps folder names clean
    return hasher.hexdigest()[:16]

# reading MD files
def build_retriever(md_paths:List[str]):
   
   """Builds a retriever for Markdown documents by processing them through multiple stages.

    This function reads Markdown files, sanitizes text for better chunking, splits documents 
    based on headers for logical grouping, applies recursive splitting to control chunk sizes, 
    and finally embeds and vectorizes the chunks to create a reranking retriever. This pipeline 
    ensures high-quality, context-preserving chunks that improve retrieval accuracy in the RAG system.

    Why these steps are used:
    - Text reading and sanitization: Ensures clean, properly formatted text to prevent embedding errors 
      and improve chunk quality by fixing spacing issues (e.g., punctuation and case transitions).
    - Header-based splitting: Preserves document structure and logical sections, making retrieval more 
      predictable and aligned with document hierarchy.
    - Recursive character splitting: Caps chunk sizes (600 chars with 100 overlap) to fit embedding 
      models' token limits while maintaining context continuity.
    - Embedding and vectorization: Converts text chunks into vectors using Ollama embeddings for 
      semantic search, stored in Chroma with cosine similarity.
    - Reranking retriever: Uses a cross-encoder to rerank top candidates, reducing duplicates and 
      improving relevance with MMR (top k=5).

    Args:
        md_paths: A list of local filesystem paths to .md files.

    Returns:
        A ContextualCompressionRetriever for efficient, high-quality document retrieval.
        
    Note:
        Empty files are skipped to avoid 'empty vector' errors. The pipeline follows a similar 
        ingestion approach as for PDF documents but adapted for Markdown's structured format.
    """
  
    # Documents ready to be vectorized
   docs: List[Document] = []
   dataset_id = compute_dataset_hash(md_paths)
  # persist dir per dataset (subfolder)
   persist_dir = CHROMA_DIR / dataset_id
    # NEW: If directory exists and has files, skip reading/splitting/sanitizing
   if persist_dir.exists() and any(persist_dir.iterdir()):
        print(f"--- Fast Loading Existing Vector Index: {dataset_id} ---")
        return embed_vectorize([], persist_dir=persist_dir)

   # Otherwise, proceed with full ingestion
   persist_dir.mkdir(parents=True, exist_ok=True)
   # iterating over md file paths and reading text  
   for p in sorted(md_paths):
        path=Path(p)
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        clean_text = sanitize_text(text)
        if not clean_text:
            continue
        docs.append(Document(
            page_content=clean_text, 
            metadata={"source": str(p), 
            "filename":Path(p).name,
            "doc_id": Path(p).stem
            }
        ))


   # splitting the document to chunks based on headers
   split_docs=split_by_md(docs)
   
   #recusrive splitting for hugh chunk of MD chunks
   recur_split= cap_chunk_size(split_docs)
   
   #embed and vectorized chunks
   retriever= embed_vectorize(recur_split,persist_dir=persist_dir)    
   
   return retriever


# duouble text sanitizing for bettter chunking
def sanitize_text(text: str) -> str:
    """
    Sanitizes text by fixing common spacing issues that can degrade embedding quality.

    Why these fixes are used:
    - Insert space after punctuation followed by letters: Prevents tokenization errors where punctuation 
      sticks to words (e.g., 'inputs.##' becomes 'inputs. ##'), ensuring proper word boundaries.
    - Add space between lowercase and uppercase: Separates camelCase or PascalCase words for better 
      semantic understanding (e.g., 'Customerchurn' becomes 'Customer churn').
    - Fix math/symbol collapse: Ensures operators like '=' are properly spaced from variables 
      (e.g., 'Revenue=Price' becomes 'Revenue = Price'), improving parsing in technical content.
    - Collapse extra whitespace: Normalizes multiple spaces/tabs/newlines to single spaces for 
      consistent chunking and embedding input.

    Args:
        text (str): The raw text to sanitize.

    Returns:
        str: The sanitized text with improved spacing for better embedding quality.
    """
    # 1. Insert space after punctuation if followed by a letter (Fixes 'inputs.##')
    text = re.sub(r'([:.!?])([a-zA-Z])', r'\1 \2', text)
    
    # 2. Add space between lowercase and uppercase (Fixes 'Customerchurn')
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # 3. Fix math/symbol collapse (Fixes 'Revenue=Price')
    text = re.sub(r'([a-zA-Z])([=*])', r'\1 \2', text)
    text = re.sub(r'([=*])([a-zA-Z])', r'\1 \2', text)
    
    # 4. Collapse extra whitespace back to single spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()




def split_by_md (docs:List[Document])-> List[Document]:
    """
    Splits Markdown documents into logical sections based on header hierarchy to preserve document structure and improve retrieval quality.

    This function preprocesses text to ensure proper header recognition, uses MarkdownHeaderTextSplitter for splitting, and maintains metadata inheritance for traceability.

    Args:
        docs: List of Document objects from Markdown files.

    Returns:
        List of Document chunks split by headers, with inherited and header-specific metadata.
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#","h1"),
            ("##","h2"),
            ("###","h3"),
            ("####","h4")
        ],
        strip_headers=False
    )

    out: List[Document]= []

    for d in docs:
       # 1. Remove ANY leading whitespace from lines starting with '#'
        fixed_content = re.sub(r'^[ \t]+#', '#', d.page_content, flags=re.MULTILINE)
        
        # 2. Ensure there is a newline BEFORE every '##' so the splitter sees it
        fixed_content = re.sub(r'(##+)', r'\n\1', fixed_content)
        
        # 3. Ensure there is a space AFTER the '#' symbols (e.g., '##Header' -> '## Header')
        fixed_content = re.sub(r'^(#+)([^#\s])', r'\1 \2', fixed_content, flags=re.MULTILINE)

        sections = splitter.split_text(fixed_content)
        # Split text into header-based chunks
        for s in sections:
            # Metadata Inheritance: Cleanly merge parent info with header info
            # We ensure we don't carry the 'whole text' in metadata
            s.metadata = {
                "source": d.metadata["source"],
                "filename": d.metadata["filename"],
                "doc_id": d.metadata["doc_id"],
                **s.metadata # Adds h1, h2, h3
            }
        
        out.extend(sections)
    
    return out


#  for size controlling for markdown chunks

def cap_chunk_size(docs: List[Document]) -> List[Document]:
    """
    Limits chunk sizes to fit embedding model constraints while preserving context through overlaps.

    This function subdivides oversized sections from header-based splits, ensuring no chunk exceeds the limit and maintains continuity with overlaps.

    Args:
        docs: List of Documents already logically split by Markdown headers.

    Returns:
        List of Documents where no chunk exceeds the size limit.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,         # chunk size of splitted text roughly 200 tokens
        chunk_overlap=100,      # overlap to prevent losing context at the edges
        length_function=len,    # Standard character-based length
        add_start_index=True    # Helps with 'traceability' to original file
    )

    # split_documents is better than split_text here because it 
    # AUTOMATICALLY propagates the metadata (h1, h2, source) to every sub-chunk.
    return splitter.split_documents(docs)


# Embedding and Vectorization
def embed_vectorize(chunks: List[Document],persist_dir: Path,force_rebuild:bool=False):
    """
    Embeds document chunks into vectors, stores them in Chroma database, and returns a reranking retriever for efficient, high-quality retrieval.

    This function handles vector database creation/loading, embedding generation, and retriever setup with cross-encoder reranking to improve relevance.

    Args:
        chunks: List of Document chunks to embed and store.
        force_rebuild: If True, rebuilds the vector database from scratch.

    Returns:
        ContextualCompressionRetriever for retrieving relevant documents with reranking.
    """
    
    # rebuild only this dataset’s folder
    if force_rebuild and persist_dir.exists():
        print(f"Force-Rebuilding Vector DB at {persist_dir}")
        shutil.rmtree(persist_dir)

    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    persist_dir_str = str(persist_dir)

    
    # load/build using persist_dir (NOT CHROMA_DIR)
    if persist_dir.exists() and any(persist_dir.iterdir()):
        print(f"--- Loading existing Vector Store from {persist_dir_str} ---")
        db = Chroma(
            persist_directory=persist_dir_str,
            embedding_function=embeddings,
            collection_name=COLLECTION,
            collection_metadata={"hnsw:space": "cosine"}
        )
    else:
        print(f"--- Vector Store not found. Building new index at {persist_dir_str} ---")
        persist_dir.mkdir(parents=True, exist_ok=True)
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION,
            persist_directory=persist_dir_str,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Successfully vectorized {len(chunks)} chunks.")

    # Create Base Retriever (The "Wide Net")
    # We fetch 20 documents instead of 5 to ensure we don't miss anything.
    base_retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20} 
    )

    # Initialize Reranker (The "Grader")
    # 'ms-marco-MiniLM-L-6-v2' is small, fast, and excellent for relevance ranking.
    # It runs locally on your Mac CPU/GPU.
    print("--- Initializing Local Reranker ---")
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # We tell it to pick the Top 5 winners from the 20 candidates
    compressor = CrossEncoderReranker(model=model, top_n=5)

    # Create Compression Retriever 
    # This wraps the base retriever. When you call .invoke(), it does the 2-step process automatically.
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever