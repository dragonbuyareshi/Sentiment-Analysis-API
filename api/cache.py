"""Redis cache layer — TTL=1hr. Falls back gracefully if Redis is down."""

import hashlib, json, os
from typing import Optional, Dict
import redis.asyncio as aioredis
from loguru import logger

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
TTL       = int(os.getenv("CACHE_TTL", 3600))  # 1 hour


class CacheLayer:
    def __init__(self):
        self.client = None
        self.connected = False

    async def connect(self):
        try:
            self.client = aioredis.from_url(REDIS_URL, decode_responses=True)
            await self.client.ping()
            self.connected = True
            logger.info(f"✔ Redis connected — {REDIS_URL}")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}) — cache disabled")
            self.connected = False

    async def disconnect(self):
        if self.client:
            await self.client.close()

    def _key(self, text: str) -> str:
        h = hashlib.sha256(text.encode()).hexdigest()[:20]
        return f"sentiment:v2:{h}"

    async def get(self, text: str) -> Optional[Dict]:
        if not self.connected: return None
        try:
            raw = await self.client.get(self._key(text))
            return json.loads(raw) if raw else None
        except:
            return None

    async def set(self, text: str, value: Dict):
        if not self.connected: return
        try:
            await self.client.setex(self._key(text), TTL, json.dumps(value))
        except:
            pass