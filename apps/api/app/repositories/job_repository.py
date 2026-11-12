from threading import Lock
from uuid import UUID

from app.models.job import ProcessingJob


class JobRepository:
    def __init__(self) -> None:
        self._jobs: dict[UUID, ProcessingJob] = {}
        self._lock = Lock()

    def add(
        self,
        job: ProcessingJob,
    ) -> ProcessingJob:
        with self._lock:
            if job.id in self._jobs:
                raise ValueError(
                    f"Job already exists: {job.id}"
                )

            self._jobs[job.id] = job

        return job

    def get(
        self,
        job_id: UUID,
    ) -> ProcessingJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_all(self) -> list:
        with self._lock:
            return sorted(
                self._jobs.values(),
                key=lambda job: job.created_at,
                reverse=True,
            )

    def remove(
        self,
        job_id: UUID,
    ) -> ProcessingJob | None:
        with self._lock:
            return self._jobs.pop(job_id, None)

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()


job_repository = JobRepository()