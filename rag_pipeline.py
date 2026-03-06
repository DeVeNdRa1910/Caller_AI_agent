"""
rag_pipeline.py — Multi-tenant RAG using FAISS + pickle (no ChromaDB).

WHY FAISS instead of ChromaDB:
  ChromaDB depends on tokenizers (Rust/pyo3) which only supports up to Python 3.13.
  FAISS is a pure C++ library with a stable Python 3.14-compatible wheel.
  Performance is identical or better: FAISS is used in production by Meta/OpenAI.

ARCHITECTURE:
  - Per-tenant FAISS IndexFlatIP (inner product on normalised vectors = cosine similarity)
  - Metadata + raw chunks stored in parallel pickle file alongside the FAISS index
  - asyncio.to_thread wraps all blocking FAISS/numpy/disk ops — never blocks event loop
  - Embedding model loaded once at startup, shared across all tenants

PERSISTENCE layout (FAISS_PERSIST_DIR):
  <dir>/
    tenant_<id>.index      <- FAISS binary index
    tenant_<id>.meta.pkl   <- list of {id, doc_id, source, chunk_index, text}

DEPENDENCIES:
  pip install faiss-cpu sentence-transformers pypdf python-multipart
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import pickle
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# -- Config --------------------------------------------------------------------
FAISS_PERSIST_DIR    = os.getenv("FAISS_PERSIST_DIR", "./faiss_db")
EMBED_MODEL_NAME     = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

CHUNK_SIZE           = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP        = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K_DEFAULT        = int(os.getenv("RAG_TOP_K", "4"))
MAX_CONTEXT_CHARS    = int(os.getenv("RAG_MAX_CONTEXT", "1200"))
SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIM_THRESHOLD", "0.35"))

EMBED_DIM = 384   # paraphrase-multilingual-MiniLM-L12-v2 output dim

os.makedirs(FAISS_PERSIST_DIR, exist_ok=True)

# -- Lazy globals --------------------------------------------------------------
_embed_model = None
_tenant_state: dict[str, dict] = {}
_tenant_locks: dict[str, asyncio.Lock] = {}
_global_lock = asyncio.Lock()


# -- File paths ----------------------------------------------------------------

def _safe_name(tenant_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", tenant_id)[:50]

def _index_path(tenant_id: str) -> str:
    return os.path.join(FAISS_PERSIST_DIR, f"tenant_{_safe_name(tenant_id)}.index")

def _meta_path(tenant_id: str) -> str:
    return os.path.join(FAISS_PERSIST_DIR, f"tenant_{_safe_name(tenant_id)}.meta.pkl")


# -- Startup -------------------------------------------------------------------

async def rag_startup():
    """Load embedding model. Call once at FastAPI startup."""
    log.info("RAG: loading embedding model '%s'...", EMBED_MODEL_NAME)
    t0 = time.perf_counter()
    await asyncio.to_thread(_load_embed_model)
    log.info("RAG: embedding model ready (%.2fs)", time.perf_counter() - t0)
    log.info("RAG: FAISS store ready at '%s'", FAISS_PERSIST_DIR)


def _load_embed_model():
    global _embed_model
    from sentence_transformers import SentenceTransformer
    _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    _embed_model.encode(["warmup"], show_progress_bar=False, normalize_embeddings=True)


# -- Tenant state management ---------------------------------------------------

async def _get_tenant_lock(tenant_id: str) -> asyncio.Lock:
    async with _global_lock:
        if tenant_id not in _tenant_locks:
            _tenant_locks[tenant_id] = asyncio.Lock()
        return _tenant_locks[tenant_id]


def _load_tenant_state_sync(tenant_id: str) -> dict:
    """Load FAISS index + metadata from disk. Returns empty state if not found."""
    import faiss
    idx_path  = _index_path(tenant_id)
    meta_path = _meta_path(tenant_id)

    if os.path.isfile(idx_path) and os.path.isfile(meta_path):
        try:
            index = faiss.read_index(idx_path)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            log.debug("Loaded FAISS index for tenant=%s (%d vectors)", tenant_id, index.ntotal)
            return {"index": index, "meta": meta}
        except Exception as e:
            log.warning("Failed to load FAISS state for tenant=%s: %s — starting fresh", tenant_id, e)

    index = faiss.IndexFlatIP(EMBED_DIM)
    return {"index": index, "meta": []}


def _save_tenant_state_sync(tenant_id: str, state: dict):
    import faiss
    faiss.write_index(state["index"], _index_path(tenant_id))
    with open(_meta_path(tenant_id), "wb") as f:
        pickle.dump(state["meta"], f)


async def _get_state(tenant_id: str) -> dict:
    if tenant_id not in _tenant_state:
        state = await asyncio.to_thread(_load_tenant_state_sync, tenant_id)
        _tenant_state[tenant_id] = state
    return _tenant_state[tenant_id]


# -- Text splitting ------------------------------------------------------------

def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    separators = ["\n\n", "\n", "। ", ". ", "! ", "? ", " ", ""]
    chunks = _recursive_split(text.strip(), separators, chunk_size, overlap)
    return [c for c in chunks if c.strip()]


def _recursive_split(text: str, separators: list[str], size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text] if text.strip() else []

    sep = ""
    new_seps: list[str] = []
    for i, s in enumerate(separators):
        if s and s in text:
            sep = s
            new_seps = separators[i + 1:]
            break

    splits  = text.split(sep) if sep else list(text)
    chunks: list[str] = []
    current = ""

    for part in splits:
        part = part.strip()
        if not part:
            continue
        candidate = (current + sep + part).strip() if current else part
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > size:
                sub = _recursive_split(part, new_seps, size, overlap)
                chunks.extend(sub)
                current = sub[-1] if sub else ""
            else:
                current = part

    if current:
        chunks.append(current)

    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append((prev_tail + " " + chunks[i]).strip())
        return overlapped

    return chunks


# -- Embedding helper ----------------------------------------------------------

def _embed_sync(texts: list[str]) -> np.ndarray:
    vecs = _embed_model.encode(
        texts,
        show_progress_bar=False,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


# -- Document ingestion --------------------------------------------------------

async def ingest_text(
    tenant_id: str,
    text: str,
    source_name: str = "document",
    doc_id: Optional[str] = None,
) -> dict:
    if not _embed_model:
        raise RuntimeError("RAG not initialised — call rag_startup() first")

    t0     = time.perf_counter()
    doc_id = doc_id or hashlib.sha256(text.encode()).hexdigest()[:16]
    chunks = _split_text(text)
    if not chunks:
        return {"tenant_id": tenant_id, "doc_id": doc_id, "chunks_added": 0, "source": source_name}

    embeddings = await asyncio.to_thread(_embed_sync, chunks)

    lock = await _get_tenant_lock(tenant_id)
    async with lock:
        state = await _get_state(tenant_id)

        new_meta = [
            {
                "id":          f"{doc_id}_chunk_{i}",
                "doc_id":      doc_id,
                "source":      source_name,
                "chunk_index": i,
                "text":        chunks[i],
            }
            for i in range(len(chunks))
        ]

        state["index"].add(embeddings)
        state["meta"].extend(new_meta)
        await asyncio.to_thread(_save_tenant_state_sync, tenant_id, state)

    log.info(
        "RAG ingest: tenant=%s doc=%s source=%s chunks=%d (%.3fs)",
        tenant_id, doc_id, source_name, len(chunks), time.perf_counter() - t0,
    )
    return {"tenant_id": tenant_id, "doc_id": doc_id, "chunks_added": len(chunks), "source": source_name}


async def ingest_pdf(tenant_id: str, pdf_bytes: bytes, filename: str = "document.pdf") -> dict:
    text = await asyncio.to_thread(_extract_pdf_text, pdf_bytes)
    if not text.strip():
        raise ValueError("PDF appears to be empty or image-only (no extractable text)")
    doc_id = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    return await ingest_text(tenant_id, text, source_name=filename, doc_id=doc_id)


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    import io
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n\n".join(p.extract_text() or "" for p in reader.pages)


async def ingest_file_bytes(
    tenant_id: str,
    file_bytes: bytes,
    filename: str,
    content_type: str = "",
) -> dict:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf" or "pdf" in content_type:
        return await ingest_pdf(tenant_id, file_bytes, filename)
    try:
        text = file_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        raise ValueError(f"Cannot decode file '{filename}': {e}")
    return await ingest_text(tenant_id, text, source_name=filename)


# -- Retrieval -----------------------------------------------------------------

async def retrieve_context(
    tenant_id: str,
    query: str,
    top_k: int = TOP_K_DEFAULT,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Embed query -> FAISS cosine search -> return context string.
    Latency: ~15-30ms (embed ~10ms + FAISS <1ms).
    """
    if not _embed_model:
        return ""

    t0 = time.perf_counter()

    try:
        lock = await _get_tenant_lock(tenant_id)
        async with lock:
            state = await _get_state(tenant_id)

        if state["index"].ntotal == 0:
            return ""

        def _search():
            q_vec = _embed_sync([query])
            k = min(top_k, state["index"].ntotal)
            scores, indices = state["index"].search(q_vec, k)
            return scores[0].tolist(), indices[0].tolist()

        scores, indices = await asyncio.to_thread(_search)

        parts: list[str] = []
        total = 0
        seen: set[str] = set()

        for score, idx in zip(scores, indices):
            if idx < 0 or score < SIMILARITY_THRESHOLD:
                continue
            entry = state["meta"][idx]
            text  = entry["text"]
            sig   = text[:80]
            if sig in seen:
                continue
            seen.add(sig)

            if total + len(text) > max_chars:
                remaining = max_chars - total
                if remaining > 80:
                    parts.append(text[:remaining])
                break
            parts.append(text)
            total += len(text)

        if not parts:
            return ""

        log.info("RAG retrieve: tenant=%s chunks=%d score_top=%.3f (%.3fs)",
                 tenant_id, len(parts), scores[0] if scores else 0, time.perf_counter() - t0)
        return "\n---\n".join(parts)

    except Exception as e:
        log.warning("RAG retrieve error: %s", e)
        return ""


# -- Document management -------------------------------------------------------

async def list_tenant_documents(tenant_id: str) -> list[dict]:
    try:
        lock = await _get_tenant_lock(tenant_id)
        async with lock:
            state = await _get_state(tenant_id)
        docs: dict[str, dict] = {}
        for entry in state["meta"]:
            did = entry["doc_id"]
            if did not in docs:
                docs[did] = {"doc_id": did, "source": entry["source"], "chunk_count": 0}
            docs[did]["chunk_count"] += 1
        return list(docs.values())
    except Exception as e:
        log.warning("RAG list_docs error: %s", e)
        return []


async def delete_tenant_documents(tenant_id: str, doc_id: str) -> int:
    import faiss

    lock = await _get_tenant_lock(tenant_id)
    async with lock:
        state    = await _get_state(tenant_id)
        old_meta = state["meta"]
        keep     = [i for i, m in enumerate(old_meta) if m["doc_id"] != doc_id]
        removed  = len(old_meta) - len(keep)

        if removed == 0:
            return 0

        if keep:
            def _rebuild():
                old_index = state["index"]
                vecs = np.vstack([old_index.reconstruct(i) for i in keep]).astype(np.float32)
                new_index = faiss.IndexFlatIP(EMBED_DIM)
                new_index.add(vecs)
                return new_index
            new_index = await asyncio.to_thread(_rebuild)
            new_meta  = [old_meta[i] for i in keep]
        else:
            new_index = faiss.IndexFlatIP(EMBED_DIM)
            new_meta  = []

        state["index"] = new_index
        state["meta"]  = new_meta
        await asyncio.to_thread(_save_tenant_state_sync, tenant_id, state)

    log.info("RAG delete: tenant=%s doc=%s chunks_removed=%d", tenant_id, doc_id, removed)
    return removed


async def delete_tenant_collection(tenant_id: str) -> bool:
    lock = await _get_tenant_lock(tenant_id)
    async with lock:
        _tenant_state.pop(tenant_id, None)
        for path in (_index_path(tenant_id), _meta_path(tenant_id)):
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError as e:
                log.warning("Could not remove %s: %s", path, e)
    log.info("RAG: deleted collection for tenant=%s", tenant_id)
    return True


# -- Context injection helper --------------------------------------------------

def build_rag_system_prompt(base_system_prompt: str, context: str) -> str:
    """
    Inject retrieved context into system prompt.

    - Context found     → inject as KNOWLEDGE BASE section agent must use.
    - No context found  → inject a clear warning so agent says "I don't know"
                          instead of hallucinating from training data.
    """
    if context.strip():
        knowledge_section = f"""
════════════════════════════════════════
KNOWLEDGE BASE  (your only source of facts)
════════════════════════════════════════
The following information was retrieved from the company's documents.
Answer the user's question using ONLY this content.
If the answer is not here, say you don't have the information.

{context}
════════════════════════════════════════
"""
    else:
        knowledge_section = """
════════════════════════════════════════
KNOWLEDGE BASE  (your only source of facts)
════════════════════════════════════════
No relevant document content was found for this query.
Do NOT answer from general knowledge.
Tell the user honestly:
  Hindi:   "मुझे इस बारे में अभी जानकारी नहीं है। मैं टीम से पता करके बताऊंगी।"
  English: "I don't have that information right now. I'll check with the team."
════════════════════════════════════════
"""
    return base_system_prompt + knowledge_section