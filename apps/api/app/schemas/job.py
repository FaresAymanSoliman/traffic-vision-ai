from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.job import JobStatus, ProcessingJob


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: JobStatus
    original_filename: str
    content_type: str
    file_size_bytes: int
    progress: float = Field(
        ge=0.0,
        le=100.0,
    )
    created_at: datetime
    updated_at: datetime
    error: str | None

    @classmethod
    def from_job(
        cls,
        job: ProcessingJob,
    ) -> "JobResponse":
        return cls(
            id=job.id,
            status=job.status,
            original_filename=job.original_filename,
            content_type=job.content_type,
            file_size_bytes=job.file_size_bytes,
            progress=job.progress,
            created_at=job.created_at,
            updated_at=job.updated_at,
            error=job.error,
        )


class JobListResponse(BaseModel):
    items: list[JobResponse]
    total: int


class ApiErrorDetail(BaseModel):
    code: str
    message: str


class ApiErrorResponse(BaseModel):
    error: ApiErrorDetail