from __future__ import annotations

from typing import List

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class CapabilityResponse(BaseModel):
    service: str
    memory_mode: str
    features: List[str]
