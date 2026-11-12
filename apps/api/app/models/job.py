from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from uuid import UUID


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class ProcessingJob:
    id: UUID
    status: JobStatus
    original_filename: str
    stored_filename: str
    input_path: Path
    output_path: Path
    summary_path: Path
    content_type: str
    file_size_bytes: int
    progress: float
    created_at: datetime
    updated_at: datetime
    error: str | None = None

    @classmethod
    def create(
        cls,
        job_id: UUID,
        original_filename: str,
        stored_filename: str,
        input_path: Path,
        output_path: Path,
        summary_path: Path,
        content_type: str,
        file_size_bytes: int,
    ) -> "ProcessingJob":
        current_time = datetime.now(UTC)

        return cls(
            id=job_id,
            status=JobStatus.PENDING,
            original_filename=original_filename,
            stored_filename=stored_filename,
            input_path=input_path,
            output_path=output_path,
            summary_path=summary_path,
            content_type=content_type,
            file_size_bytes=file_size_bytes,
            progress=0.0,
            created_at=current_time,
            updated_at=current_time,
        )