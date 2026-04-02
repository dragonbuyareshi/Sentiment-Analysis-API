"""
api/auth.py
───────────
Bearer API-key authentication dependency.

Usage in endpoints:
    @app.post("/api/v1/analyze")
    async def analyze(req: AnalyzeRequest, _: str = Depends(verify_api_key)):
        ...

Keys are loaded from the API_KEYS env var (comma-separated).
In production: rotate keys via environment without restarting the service.
"""

import os
from fastapi import Header, HTTPException, status

# Load valid keys from environment at startup
# Default: dev-key-123 for local development only
_raw_keys = os.getenv("API_KEYS", "dev-key-123")
VALID_KEYS: set = {k.strip() for k in _raw_keys.split(",") if k.strip()}


async def verify_api_key(authorization: str = Header(None)) -> str:
    """
    FastAPI dependency: validates 'Authorization: Bearer <key>' header.
    Returns the validated key string on success.
    Raises HTTP 401 / 403 on failure.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing. Format: Bearer <api-key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization format. Expected: Bearer <api-key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    key = parts[1]
    if key not in VALID_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return key
