"""
tenant_manager.py — Multi-tenant registry with API key authentication.

Stores tenant metadata in a JSON file (simple, no extra DB dependency).
In production, replace with Postgres/Redis.

Schema per tenant:
  {
    "tenant_id":   "acme_logistics",
    "name":        "Acme Logistics",
    "api_key":     "sk_acme_xxxx",        ← hashed in storage
    "created_at":  "2024-01-01T00:00:00",
    "agent_name":  "SONY",               ← overrides default agent name
    "system_prompt_override": null,       ← full override or None to use base
    "extra_context": "",                  ← static context always injected (e.g. pricing table)
    "language_preference": "hi-IN",
    "active": true
  }
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

TENANT_DB_PATH = os.getenv("TENANT_DB_PATH", "./tenant_db.json")
_lock = asyncio.Lock()


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def _load_db() -> dict:
    p = Path(TENANT_DB_PATH)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _save_db(db: dict):
    Path(TENANT_DB_PATH).write_text(json.dumps(db, indent=2, ensure_ascii=False))


# ── Public API ─────────────────────────────────────────────────────────────────

async def create_tenant(
    tenant_id: str,
    name: str,
    agent_name: str = "SONY",
    language_preference: str = "hi-IN",
    system_prompt_override: Optional[str] = None,
    extra_context: str = "",
) -> dict:
    """
    Register a new tenant. Returns {tenant_id, api_key, ...}.
    The raw api_key is returned ONCE — store it securely.
    """
    async with _lock:
        db = await asyncio.to_thread(_load_db)
        if tenant_id in db:
            raise ValueError(f"Tenant '{tenant_id}' already exists")

        raw_key = f"sk_{tenant_id}_{secrets.token_urlsafe(24)}"
        record  = {
            "tenant_id":              tenant_id,
            "name":                   name,
            "api_key_hash":           _hash_key(raw_key),
            "created_at":             datetime.now(timezone.utc).isoformat(),
            "agent_name":             agent_name,
            "system_prompt_override": system_prompt_override,
            "extra_context":          extra_context,
            "language_preference":    language_preference,
            "active":                 True,
        }
        db[tenant_id] = record
        await asyncio.to_thread(_save_db, db)

        log.info("Tenant created: %s (%s)", tenant_id, name)
        return {**record, "api_key": raw_key}   # return raw key only on creation


async def get_tenant(tenant_id: str) -> Optional[dict]:
    db = await asyncio.to_thread(_load_db)
    return db.get(tenant_id)


async def get_tenant_by_api_key(api_key: str) -> Optional[dict]:
    """Authenticate by API key. Returns tenant record or None."""
    hashed = _hash_key(api_key)
    db     = await asyncio.to_thread(_load_db)
    for record in db.values():
        if record.get("api_key_hash") == hashed and record.get("active"):
            return record
    return None


async def list_tenants() -> list[dict]:
    db = await asyncio.to_thread(_load_db)
    return [
        {k: v for k, v in t.items() if k != "api_key_hash"}
        for t in db.values()
    ]


async def update_tenant(tenant_id: str, **fields) -> Optional[dict]:
    """Update mutable fields: agent_name, extra_context, system_prompt_override, active."""
    allowed = {"agent_name", "extra_context", "system_prompt_override", "language_preference", "active", "name"}
    async with _lock:
        db = await asyncio.to_thread(_load_db)
        if tenant_id not in db:
            return None
        for k, v in fields.items():
            if k in allowed:
                db[tenant_id][k] = v
        await asyncio.to_thread(_save_db, db)
        return {k: v for k, v in db[tenant_id].items() if k != "api_key_hash"}


async def delete_tenant(tenant_id: str) -> bool:
    async with _lock:
        db = await asyncio.to_thread(_load_db)
        if tenant_id not in db:
            return False
        del db[tenant_id]
        await asyncio.to_thread(_save_db, db)
    log.info("Tenant deleted: %s", tenant_id)
    return True


async def rotate_api_key(tenant_id: str) -> Optional[str]:
    """Generate a new API key for a tenant. Returns new raw key or None if tenant not found."""
    async with _lock:
        db = await asyncio.to_thread(_load_db)
        if tenant_id not in db:
            return None
        raw_key = f"sk_{tenant_id}_{secrets.token_urlsafe(24)}"
        db[tenant_id]["api_key_hash"] = _hash_key(raw_key)
        await asyncio.to_thread(_save_db, db)
    log.info("API key rotated for tenant: %s", tenant_id)
    return raw_key