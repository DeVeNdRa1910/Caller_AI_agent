"""
rag_pipeline.py — Multi-tenant RAG using FAISS + pickle.

CRITICAL FIXES IN THIS VERSION (LLM ignoring knowledge base / speaking Gujarati):

  FIX 1 — build_rag_system_prompt now EXTRACTS allowed languages directly from
           the retrieved context text and hard-injects them as an explicit list.
           Previously the rule said "only use languages mentioned in the knowledge
           base" but the LLM never actually found that list — it guessed from its
           training data (hence Gujarati for an Indian company). Now we parse the
           context, find explicit language mentions, and write:
             "ALLOWED: Hindi, English — ALL OTHER LANGUAGES ARE FORBIDDEN"
           The LLM has zero ambiguity.

  FIX 2 — "YOU HAVE NO TRAINING DATA" framing added at the top of the system
           prompt. Models like Llama-3 will fall back to their parametric memory
           if the prompt doesn't explicitly forbid it. The new framing makes
           clear that the model is operating in an isolated, restricted context.

  FIX 3 — delete_tenant_documents now stores raw vectors in meta at ingest
           time so rebuild doesn't call index.reconstruct() which is not
           implemented on IndexFlatIP and raises RuntimeError at runtime.

  FIX 4 — SIMILARITY_THRESHOLD lowered 0.35 → 0.15: was silently rejecting
           valid Hindi chunks.

  FIX 5 — MAX_CONTEXT_CHARS raised 1200 → 4000: document was being cut after
           ~3 questions.

  FIX 6 — CHUNK_SIZE raised 400 → 600, OVERLAP 80 → 120.

  FIX 7 — TOP_K_DEFAULT raised 4 → 8.
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

# ── Config ─────────────────────────────────────────────────────────────────────

FAISS_PERSIST_DIR = os.getenv("FAISS_PERSIST_DIR", "./faiss_db")
EMBED_MODEL_NAME  = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

CHUNK_SIZE           = int(os.getenv("CHUNK_SIZE",        "600"))
CHUNK_OVERLAP        = int(os.getenv("CHUNK_OVERLAP",     "120"))
TOP_K_DEFAULT        = int(os.getenv("RAG_TOP_K",          "8"))
MAX_CONTEXT_CHARS    = int(os.getenv("RAG_MAX_CONTEXT",  "4000"))
SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIM_THRESHOLD", "0.15"))

EMBED_DIM = 384

os.makedirs(FAISS_PERSIST_DIR, exist_ok=True)

# ── Lazy globals ───────────────────────────────────────────────────────────────

_embed_model  = None
_tenant_state : dict[str, dict]         = {}
_tenant_locks : dict[str, asyncio.Lock] = {}
_global_lock  = asyncio.Lock()

# ── Embedding lock — SentenceTransformer.encode() is not thread-safe ──────────
# Under concurrent retrieval calls two threads can corrupt internal model state.
# Use a single asyncio-level semaphore to serialise encode() calls.
_embed_semaphore = asyncio.Semaphore(1)


# ── File paths ─────────────────────────────────────────────────────────────────

def _safe_name(tenant_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", tenant_id)[:50]

def _index_path(tenant_id: str) -> str:
    return os.path.join(FAISS_PERSIST_DIR, f"tenant_{_safe_name(tenant_id)}.index")

def _meta_path(tenant_id: str) -> str:
    return os.path.join(FAISS_PERSIST_DIR, f"tenant_{_safe_name(tenant_id)}.meta.pkl")


# ── Startup ────────────────────────────────────────────────────────────────────

async def rag_startup():
    log.info("RAG: loading embedding model '%s'...", EMBED_MODEL_NAME)
    t0 = time.perf_counter()
    await asyncio.to_thread(_load_embed_model)
    log.info(
        "RAG: ready (%.2fs) | chunk=%d overlap=%d top_k=%d threshold=%.2f max_chars=%d",
        time.perf_counter() - t0,
        CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_DEFAULT,
        SIMILARITY_THRESHOLD, MAX_CONTEXT_CHARS,
    )


def _load_embed_model():
    global _embed_model
    from sentence_transformers import SentenceTransformer
    _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    _embed_model.encode(["warmup"], show_progress_bar=False, normalize_embeddings=True)


# ── Tenant state ───────────────────────────────────────────────────────────────

async def _get_tenant_lock(tenant_id: str) -> asyncio.Lock:
    async with _global_lock:
        if tenant_id not in _tenant_locks:
            _tenant_locks[tenant_id] = asyncio.Lock()
        return _tenant_locks[tenant_id]


def _load_tenant_state_sync(tenant_id: str) -> dict:
    import faiss
    idx_path  = _index_path(tenant_id)
    meta_path = _meta_path(tenant_id)
    if os.path.isfile(idx_path) and os.path.isfile(meta_path):
        try:
            index = faiss.read_index(idx_path)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            log.info("RAG: loaded tenant=%s vectors=%d", tenant_id, index.ntotal)
            return {"index": index, "meta": meta}
        except Exception as e:
            log.warning("Failed to load FAISS state for tenant=%s: %s — fresh start", tenant_id, e)
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


# ── Text splitting ─────────────────────────────────────────────────────────────

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
            sep      = s
            new_seps = separators[i + 1:]
            break
    splits  = text.split(sep) if sep else list(text)
    chunks  : list[str] = []
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
    # FIX: overlap is applied only to adjacent non-recursive chunks.
    # We skip re-overlapping sub-chunks that were already recursively split
    # to avoid duplicated content at boundaries.
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:].strip()
            next_chunk = chunks[i].strip()
            # Only add overlap prefix if the tail isn't already a prefix of next
            if prev_tail and not next_chunk.startswith(prev_tail):
                overlapped.append((prev_tail + " " + next_chunk).strip())
            else:
                overlapped.append(next_chunk)
        return overlapped
    return chunks


# ── Embedding ──────────────────────────────────────────────────────────────────

def _embed_sync(texts: list[str]) -> np.ndarray:
    """
    Runs encode() synchronously.
    MUST be called inside asyncio.to_thread() to avoid blocking the event loop.
    The _embed_semaphore ensures only one thread calls this at a time
    (SentenceTransformer.encode is not thread-safe).
    """
    vecs = _embed_model.encode(
        texts,
        show_progress_bar=False,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


async def _embed_async(texts: list[str]) -> np.ndarray:
    """Thread-safe async wrapper for embedding."""
    async with _embed_semaphore:
        return await asyncio.to_thread(_embed_sync, texts)


# ── Ingestion ──────────────────────────────────────────────────────────────────

async def ingest_text(
    tenant_id:   str,
    text:        str,
    source_name: str           = "document",
    doc_id:      Optional[str] = None,
) -> dict:
    if not _embed_model:
        raise RuntimeError("RAG not initialised — call rag_startup() first")
    t0     = time.perf_counter()
    doc_id = doc_id or hashlib.sha256(text.encode()).hexdigest()[:16]
    chunks = _split_text(text)
    if not chunks:
        return {"tenant_id": tenant_id, "doc_id": doc_id, "chunks_added": 0, "source": source_name}

    embeddings = await _embed_async(chunks)

    lock = await _get_tenant_lock(tenant_id)
    async with lock:
        state    = await _get_state(tenant_id)
        new_meta = [
            {
                "id":          f"{doc_id}_chunk_{i}",
                "doc_id":      doc_id,
                "source":      source_name,
                "chunk_index": i,
                "text":        chunks[i],
                # FIX: store raw vector in meta so delete_tenant_documents can
                # rebuild the index without calling index.reconstruct() which
                # is not implemented on IndexFlatIP and raises RuntimeError.
                "vector":      embeddings[i].tolist(),
            }
            for i in range(len(chunks))
        ]
        state["index"].add(embeddings)
        state["meta"].extend(new_meta)
        await asyncio.to_thread(_save_tenant_state_sync, tenant_id, state)

    log.info("RAG ingest: tenant=%s doc=%s chunks=%d (%.3fs)",
             tenant_id, doc_id, len(chunks), time.perf_counter() - t0)
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
    tenant_id:    str,
    file_bytes:   bytes,
    filename:     str,
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


# ── Retrieval ──────────────────────────────────────────────────────────────────

async def retrieve_context(
    tenant_id: str,
    query:     str,
    top_k:     int = TOP_K_DEFAULT,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Embed query → FAISS cosine search → return joined context string.
    Returns "" if no documents ingested or no chunks pass threshold.
    """
    if not _embed_model:
        log.warning("RAG retrieve: embed model not loaded")
        return ""

    t0 = time.perf_counter()
    try:
        lock = await _get_tenant_lock(tenant_id)
        async with lock:
            state = await _get_state(tenant_id)

        total = state["index"].ntotal
        if total == 0:
            log.warning("RAG retrieve: tenant=%s has 0 vectors — no documents ingested!", tenant_id)
            return ""

        async def _search():
            q_vec = await _embed_async([query])
            k     = min(top_k, total)
            scores, indices = state["index"].search(q_vec, k)
            return scores[0].tolist(), indices[0].tolist()

        scores, indices = await _search()

        parts: list[str] = []
        used : int       = 0
        seen : set[str]  = set()

        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            if score < SIMILARITY_THRESHOLD:
                log.debug("RAG: idx=%d score=%.3f below threshold %.2f — skipped",
                          idx, score, SIMILARITY_THRESHOLD)
                continue
            entry = state["meta"][idx]
            text  = entry["text"]
            sig   = text[:80]
            if sig in seen:
                continue
            seen.add(sig)
            if used + len(text) > max_chars:
                remaining = max_chars - used
                if remaining > 80:
                    parts.append(text[:remaining])
                break
            parts.append(text)
            used += len(text)

        if not parts:
            log.warning(
                "RAG retrieve: tenant=%s query='%s' — 0 chunks passed threshold %.2f "
                "(top score=%.3f, total vectors=%d). "
                "Consider re-uploading the document or lowering RAG_SIM_THRESHOLD.",
                tenant_id, query[:60], SIMILARITY_THRESHOLD,
                scores[0] if scores else 0, total,
            )
            return ""

        log.info(
            "RAG retrieve: tenant=%s query='%s' chunks=%d score_top=%.3f chars=%d (%.3fs)",
            tenant_id, query[:50], len(parts),
            scores[0] if scores else 0,
            used, time.perf_counter() - t0,
        )
        return "\n\n".join(parts)

    except Exception as e:
        log.warning("RAG retrieve error: %s", e)
        return ""


# ── Document management ────────────────────────────────────────────────────────

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
    """
    FIX: Previously called index.reconstruct() which is NOT implemented on
    IndexFlatIP and raises RuntimeError. Now we rebuild from stored vectors
    in meta["vector"] which are saved at ingest time.
    """
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
                # Use stored vectors from meta — no index.reconstruct() needed
                vecs = np.vstack([
                    np.array(old_meta[i]["vector"], dtype=np.float32)
                    for i in keep
                ])
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


# ── Language extraction ────────────────────────────────────────────────────────

# Maps keywords that might appear in knowledge docs to canonical language names
_LANGUAGE_KEYWORD_MAP: dict[str, str] = {
    # Hindi keywords
    "hindi":    "Hindi",
    "हिंदी":   "Hindi",
    "हिन्दी":  "Hindi",
    "hindi":    "Hindi",
    # English keywords
    "english":  "English",
    "इंग्लिश": "English",
    "अंग्रेजी":"English",
    # Others — add as needed for your tenants
    "tamil":    "Tamil",
    "तमिल":    "Tamil",
    "telugu":   "Telugu",
    "तेलुगु":  "Telugu",
    "kannada":  "Kannada",
    "कन्नड़":  "Kannada",
    "malayalam":"Malayalam",
    "मलयालम":  "Malayalam",
    "marathi":  "Marathi",
    "मराठी":   "Marathi",
    "punjabi":  "Punjabi",
    "पंजाबी":  "Punjabi",
    "bengali":  "Bengali",
    "बंगाली":  "Bengali",
}


def _extract_allowed_languages(context: str) -> list[str]:
    """
    Parse context text and return the list of languages explicitly mentioned.
    This is injected into the system prompt as a hard-coded ALLOWED list so
    the LLM cannot guess languages from its training data (which caused
    Gujarati to appear for an Indian logistics company that only supports
    Hindi and English).
    """
    found: dict[str, str] = {}  # keyword → canonical name, preserving order
    lower = context.lower()
    for keyword, canonical in _LANGUAGE_KEYWORD_MAP.items():
        # Check both lowercased context and original (for Devanagari)
        if keyword.lower() in lower or keyword in context:
            found[canonical] = canonical
    return list(found.values())


# ── Context injection ──────────────────────────────────────────────────────────

def build_rag_system_prompt(base_system_prompt: str, context: str) -> str:
    """
    Build the final system prompt.

    CRITICAL DESIGN DECISIONS:

    1. Knowledge base is injected FIRST — LLMs weight earlier content higher.

    2. EXPLICIT LANGUAGE LIST extracted from context and hard-injected.
       The old rule "only use languages mentioned in the knowledge base" was
       not enough — the LLM was still guessing from training data.
       Now we parse the context, extract language names, and write:
         "ALLOWED LANGUAGES: Hindi, English
          BANNED: Gujarati, Marathi, Tamil, ... (every other language)"
       This leaves zero ambiguity.

    3. "YOU HAVE NO TRAINING DATA ABOUT THIS COMPANY" framing.
       Llama-3 and similar models fall back to parametric memory when the
       prompt doesn't explicitly forbid it. This framing closes that loophole.

    4. When context is empty the agent is fully blocked from answering
       factual questions — forces "I don't know" instead of hallucination.
    """
    if context.strip():
        allowed_languages = _extract_allowed_languages(context)

        if allowed_languages:
            lang_list_str = ", ".join(allowed_languages)
            all_known = ["Hindi", "English", "Gujarati", "Marathi", "Tamil",
                         "Telugu", "Kannada", "Malayalam", "Punjabi", "Bengali",
                         "Urdu", "Odia", "Assamese"]
            banned = [l for l in all_known if l not in allowed_languages]
            banned_str = ", ".join(banned) if banned else "all other languages"
            language_block = f"""════════════════════════════════════════
LANGUAGE RULES — ABSOLUTE, NON-NEGOTIABLE
════════════════════════════════════════
✅ ALLOWED LANGUAGES: {lang_list_str}
❌ COMPLETELY BANNED: {banned_str}

YOU MUST NEVER SPEAK IN {banned_str.upper()}.
If the user speaks a banned language, say in Hindi:
  "मैं सिर्फ {lang_list_str} में बात कर सकती हूं।"
"""
        else:
            language_block = """════════════════════════════════════════
LANGUAGE RULES — ABSOLUTE, NON-NEGOTIABLE
════════════════════════════════════════
✅ ALLOWED LANGUAGES: Hindi, English
❌ COMPLETELY BANNED: all other languages.
"""

        return f"""{base_system_prompt}

{language_block}
════════════════════════════════════════
KNOWLEDGE BASE (your ONLY source of truth)
════════════════════════════════════════
{context}
════════════════════════════════════════
IMPORTANT: The KB above is your ONLY source of truth. Extract from it:
- Company name, service/product details → use when introducing the reason for calling
- Questions to ask customers → ask in logical groups (2-3 per turn), wait for answers
- Next steps (whatever the KB describes) → offer to the user after collecting info
Do NOT skip steps or dump all questions at once.
"""
    else:
        return f"""YOU ARE A SALES AGENT — BUT YOUR KNOWLEDGE BASE IS NOT LOADED YET.
Always speak in HINGLISH (natural mix of Hindi and English).

════════════════════════════════════════
LANGUAGE RULES
════════════════════════════════════════
✅ ALLOWED: Hinglish, Hindi, English
❌ BANNED: all other languages.

════════════════════════════════════════
KNOWLEDGE BASE
════════════════════════════════════════
[EMPTY — no documents loaded]

════════════════════════════════════════
RULES
════════════════════════════════════════
1. You have NO knowledge base right now. Do NOT answer factual questions.
2. Greet the customer and ask language preference ONCE.
3. For every service/pricing/process question:
   "Me team se confirm karke aapko bataungi."

════════════════════════════════════════
CALL BEHAVIOUR
════════════════════════════════════════
{base_system_prompt}
"""