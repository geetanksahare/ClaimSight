from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from claimsight.config import get_settings

api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header_scheme)) -> bool:
    """Validate X-API-Key header against configured CLAIMSIGHT_API_KEY if configured."""
    settings = get_settings()
    expected_key = settings.api_key

    # If no API key configured in env/settings, allow open access (e.g. dev/demo mode)
    if not expected_key:
        return True

    if not api_key or api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "unauthorized", "message": "Invalid or missing X-API-Key header"},
        )

    return True
