"""
rag_core.py - ChromaDB RAG system with Gemini embeddings
Supports PDF (pdfplumber), TXT, DOCX files
"""
import os
import io
import json
import logging
import asyncio
import hashlib
from typing import Optional

import chromadb
from google import genai

from database import (
    add_rag_chunk, get_rag_chunks, list_rag_docs,
    delete_rag_doc, count_user_docs
)

logger = logging.getLogger(__name__)

# Limits
MAX_FILE_SIZE = 10 * 1024 * 1024   # 10MB
MAX_DOCS_PER_USER = 20
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between chunks

# ChromaDB client (persistent)
_chroma_client = None
_chroma_collection = None


def _get_chroma():
    global _chroma_client, _chroma_collection
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path="./chroma_data")
        _chroma_collection = _chroma_client.get_or_create_collection(
            name="rag_docs",
            metadata={"hnsw:space": "cosine"}
        )
    return _chroma_collection


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


_gemini_client: genai.Client | None = None


def _get_gemini() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _gemini_client


async def _get_embedding(text: str) -> list[float]:
    """Get embedding from Gemini API."""
    def _embed():
        response = _get_gemini().models.embed_content(
            model="gemini-embedding-001",
            contents=text,
        )
        return response.embeddings[0].values

    return await asyncio.to_thread(_embed)


def _extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    import pdfplumber
    text_parts = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def _extract_text_from_docx(content: bytes) -> str:
    """Extract text from DOCX bytes using python-docx."""
    from docx import Document
    doc = Document(io.BytesIO(content))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def _extract_text(filename: str, content: bytes) -> str:
    """Extract text based on file extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return _extract_text_from_pdf(content)
    elif ext in (".docx", ".doc"):
        return _extract_text_from_docx(content)
    else:
        # TXT and other text formats
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="replace")


async def add_document(user_id: int, filename: str, content: bytes) -> str:
    """
    Add document to RAG system.
    Returns status message.
    """
    # Check file size
    if len(content) > MAX_FILE_SIZE:
        size_mb = len(content) / (1024 * 1024)
        return f"❌ File quá lớn ({size_mb:.1f}MB). Giới hạn 10MB."

    # Check doc count
    doc_count = await count_user_docs(user_id)
    if doc_count >= MAX_DOCS_PER_USER:
        return f"❌ Bạn đã đạt giới hạn {MAX_DOCS_PER_USER} tài liệu. Xóa bớt để thêm mới."

    # Extract text
    try:
        def _extract():
            return _extract_text(filename, content)
        text = await asyncio.to_thread(_extract)
    except Exception as e:
        return f"❌ Không thể đọc file: {str(e)}"

    if not text.strip():
        return "❌ File không có nội dung text."

    # Chunk text
    chunks = _chunk_text(text)
    if not chunks:
        return "❌ Không thể chia nhỏ nội dung file."

    # Embed and store
    collection = _get_chroma()
    stored = 0

    for i, chunk in enumerate(chunks):
        try:
            embedding = await _get_embedding(chunk)
            chunk_id = f"{user_id}_{hashlib.md5(f'{filename}_{i}'.encode()).hexdigest()}"

            # Store in ChromaDB
            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"user_id": str(user_id), "filename": filename, "chunk_index": i}]
            )

            # Store metadata in SQLite
            await add_rag_chunk(user_id, filename, i, chunk, json.dumps(embedding[:10]))  # store first 10 dims as preview
            stored += 1

        except Exception as e:
            logger.error(f"Failed to embed chunk {i} of {filename}: {e}")

    if stored == 0:
        return "❌ Không thể lưu tài liệu. Vui lòng thử lại."

    return (
        f"✅ Đã thêm tài liệu <b>{filename}</b>\n"
        f"📄 {len(text):,} ký tự → {stored} chunks"
    )


async def search_rag(user_id: int, query: str, top_k: int = 3) -> list[dict]:
    """Search for relevant chunks in user's documents."""
    try:
        collection = _get_chroma()
        query_embedding = await _get_embedding(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"user_id": str(user_id)}
        )

        chunks = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                distance = results["distances"][0][i] if results.get("distances") else 1.0
                chunks.append({
                    "content": doc,
                    "filename": meta.get("filename", "unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "relevance": 1 - distance,  # convert distance to relevance score
                })
        return chunks

    except Exception as e:
        logger.error(f"RAG search error for user {user_id}: {e}")
        return []


async def build_rag_context(user_id: int, query: str) -> Optional[str]:
    """Build context string from relevant RAG chunks."""
    chunks = await search_rag(user_id, query, top_k=3)
    if not chunks:
        return None

    parts = ["📚 <b>Thông tin từ tài liệu của bạn:</b>"]
    for chunk in chunks:
        if chunk["relevance"] > 0.3:  # only include reasonably relevant chunks
            parts.append(
                f"[{chunk['filename']}]:\n{chunk['content']}"
            )

    if len(parts) <= 1:
        return None

    return "\n\n".join(parts)


async def list_docs(user_id: int) -> str:
    """List user's documents."""
    docs = await list_rag_docs(user_id)
    if not docs:
        return "📭 Bạn chưa có tài liệu nào.\n\nGửi file PDF/TXT/DOCX để thêm vào RAG."

    lines = [f"📚 <b>Tài liệu của bạn ({len(docs)}/{MAX_DOCS_PER_USER}):</b>\n"]
    for i, filename in enumerate(docs, 1):
        lines.append(f"{i}. <code>{filename}</code>")

    lines.append("\n<i>Dùng /rag clear &lt;tên file&gt; để xóa</i>")
    return "\n".join(lines)


async def delete_doc(user_id: int, filename: str) -> str:
    """Delete a document from RAG."""
    # Remove from ChromaDB
    try:
        collection = _get_chroma()
        existing = collection.get(where={"user_id": str(user_id), "filename": filename})
        if existing and existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception as e:
        logger.error(f"ChromaDB delete error: {e}")

    # Remove from SQLite
    ok = await delete_rag_doc(user_id, filename)
    if ok:
        return f"🗑 Đã xóa tài liệu <b>{filename}</b>."
    return f"❌ Không tìm thấy tài liệu <b>{filename}</b>."


async def has_docs(user_id: int) -> bool:
    """Check if user has any RAG documents."""
    count = await count_user_docs(user_id)
    return count > 0
