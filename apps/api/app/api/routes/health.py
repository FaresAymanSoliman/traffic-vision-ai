from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["Health"])


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    service: str
    environment: str


@router.get("", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="traffic-vision-api",
        environment="development",
    )