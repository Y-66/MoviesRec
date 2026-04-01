from __future__ import annotations

from fastapi import APIRouter, status
from movies.api.schemas.system import CapabilityResponse, HealthResponse

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="movies-recommendation-agent",
        version="1.1.0",
    )


@router.get("/capabilities", response_model=CapabilityResponse, status_code=status.HTTP_200_OK)
async def capabilities() -> CapabilityResponse:
    return CapabilityResponse(
        service="movies-recommendation-agent",
        memory_mode="langgraph-thread-memory + file-history-for-frontend",
        features=[
            "intent-analysis",
            "conditional-sql-filter",
            "collaborative-filter-placeholder",
            "diversity-filter-placeholder",
            "response-summarization",
            "session-history-list",
            "session-history-detail",
            "session-history-delete",
        ],
    )
