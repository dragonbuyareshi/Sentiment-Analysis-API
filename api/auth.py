"""Simple API key auth via Authorization: Bearer  header."""

import os
from fastapi import Header, HTTPException, status

# In production: load from DB / secret manager
VALID_KEYS = set(os.getenv("API_KEYS", "dev-key-123,test-key-456").split(","))


async def verify_api_key(authorization: str = Header(None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Format: Bearer ")
    key = parts[1]
    if key not in VALID_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key