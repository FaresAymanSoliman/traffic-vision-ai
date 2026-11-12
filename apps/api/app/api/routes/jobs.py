from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.models.job import JobStatus, ProcessingJob
from app.repositories.job_repository import job_repository
from app.schemas.job import (
    ApiErrorDetail,
    ApiErrorResponse,
    JobListResponse,
    JobResponse,
)
from app.services.video_storage import (
    VideoStorageService,
    VideoUploadError,
)

router = APIRouter(
    prefix="/jobs",
    tags=["Jobs"],
)

settings = get_settings()
storage_service = VideoStorageService(settings)


@router.post(
    "",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "model": ApiErrorResponse,
        },
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {
            "model": ApiErrorResponse,
        },
    },
)
async def create_job(
    video: UploadFile = File(...),
) -> JobResponse:
    job_id = uuid4()

    try:
        stored_video = await storage_service.store_upload(
            upload=video,
            job_id=job_id,
        )
    except VideoUploadError as error:
        status_code = (
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            if error.code == "VIDEO_TOO_LARGE"
            else status.HTTP_400_BAD_REQUEST
        )

        raise HTTPException(
            status_code=status_code,
            detail={
                "code": error.code,
                "message": error.message,
            },
        ) from error

    job = ProcessingJob.create(
        job_id=job_id,
        original_filename=stored_video.original_filename,
        stored_filename=stored_video.stored_filename,
        input_path=stored_video.input_path,
        output_path=stored_video.output_path,
        summary_path=stored_video.summary_path,
        content_type=stored_video.content_type,
        file_size_bytes=stored_video.file_size_bytes,
    )

    job_repository.add(job)

    return JobResponse.from_job(job)


@router.get(
    "",
    response_model=JobListResponse,
)
def list_jobs() -> JobListResponse:
    jobs = job_repository.list_all()

    items = [
        JobResponse.from_job(job)
        for job in jobs
    ]

    return JobListResponse(
        items=items,
        total=len(items),
    )


@router.get(
    "/{job_id}",
    response_model=JobResponse,
)
def get_job(
    job_id: UUID,
) -> JobResponse:
    job = find_job_or_404(job_id)

    return JobResponse.from_job(job)


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_job(
    job_id: UUID,
) -> None:
    job = find_job_or_404(job_id)

    if job.status == JobStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "JOB_IS_PROCESSING",
                "message": (
                    "A processing job cannot be deleted."
                ),
            },
        )

    storage_service.delete_job_files(
        input_path=job.input_path,
        output_path=job.output_path,
    )

    job_repository.remove(job_id)


def find_job_or_404(
    job_id: UUID,
) -> ProcessingJob:
    job = job_repository.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "JOB_NOT_FOUND",
                "message": f"Job was not found: {job_id}",
            },
        )

    return job