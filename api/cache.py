"""
api/cache.py
────────────
Async Redis cache layer for sentiment predictions.

- Keys are SHA-256 hashes of the input text (first 20 hex chars)
- TTL default: 3600 seconds (1 hour)
- Gracefully disables itself if Redis is unreachable
- Thread-safe via asyncio; no locks needed
"""

import hashlib
import json
import os
from typing import Any, Dict, Optional

import redis.asyncio as aioredis
from loguru import logger

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
TTL       = int(os.getenv("CACHE_TTL", "3600"))
KEY_PREFIX = "sentiment:v2:"


class CacheLayer:
    """Async Redis cache with graceful fallback on connection failure."""

    def __init__(self) -> None:
        self.client: Optional[aioredis.Redis] = None
        self.connected: bool = False

    async def connect(self) -> None:
        """Initialize Redis connection. Safe to call multiple times."""
        try:
            self.client = aioredis.from_url(
                REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=2,
            )
            await self.client.ping()
            self.connected = True
            logger.info(f"✔ Redis connected — {REDIS_URL} (TTL={TTL}s)")
        except Exception as exc:
            self.connected = False
            logger.warning(f"Redis unavailable ({exc}) — cache disabled, running without cache")

    async def disconnect(self) -> None:
        """Gracefully close the Redis connection."""
        if self.client:
            try:
                await self.client.aclose()
            except Exception:
                pass

    # ── Internal helpers ──────────────────────────────────────────

    def _key(self, text: str) -> str:
        """Generate a stable, bounded cache key from input text."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
        return f"{KEY_PREFIX}{digest}"

    # ── Public interface ──────────────────────────────────────────

    async def get(self, text: str) -> Optional[Dict[str, Any]]:
        """Return cached prediction dict, or None on miss / Redis error."""
        if not self.connected:
            return None
        try:
            raw = await self.client.get(self._key(text))
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.debug(f"Cache GET error (non-fatal): {exc}")
            return None

    async def set(self, text: str, value: Dict[str, Any]) -> None:
        """Cache a prediction dict with TTL. Fails silently."""
        if not self.connected:
            return
        try:
            await self.client.setex(
                self._key(text),
                TTL,
                json.dumps(value, ensure_ascii=False),
            )
        except Exception as exc:
            logger.debug(f"Cache SET error (non-fatal): {exc}")

    async def delete(self, text: str) -> None:
        """Remove a specific cache entry."""
        if not self.connected:
            return
        try:
            await self.client.delete(self._key(text))
        except Exception:
            pass

    async def flush(self) -> None:
        """Flush all sentiment cache keys (use with caution)."""
        if not self.connected:
            return
        try:
            keys = await self.client.keys(f"{KEY_PREFIX}*")
            if keys:
                await self.client.delete(*keys)
                logger.info(f"Flushed {len(keys)} cache entries")
        except Exception as exc:
            logger.warning(f"Cache flush error: {exc}")

    async def info(self) -> Dict[str, Any]:
        """Return Redis server info for health checks."""
        if not self.connected:
            return {"connected": False}
        try:
            info = await self.client.info("server")
            return {
                "connected": True,
                "redis_version": info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
            }
        except Exception:
            return {"connected": False}
