from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile

from app.core.config import Settings


class VideoUploadError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
    ) -> None:
        super().__init__(message)

        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class StoredVideo:
    original_filename: str
    stored_filename: str
    input_path: Path
    output_path: Path
    summary_path: Path
    content_type: str
    file_size_bytes: int


class VideoStorageService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def store_upload(
        self,
        upload: UploadFile,
        job_id: UUID,
    ) -> StoredVideo:
        original_filename = self._validate_filename(
            upload.filename
        )

        extension = Path(original_filename).suffix.lower()

        self._validate_extension(extension)
        self._validate_content_type(upload.content_type)

        job_input_directory = (
            self._settings.input_jobs_directory
            / str(job_id)
        )

        job_output_directory = (
            self._settings.output_jobs_directory
            / str(job_id)
        )

        job_input_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        job_output_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        stored_filename = f"input{extension}"

        input_path = (
            job_input_directory / stored_filename
        )

        output_path = (
            job_output_directory / "processed.mp4"
        )

        summary_path = (
            job_output_directory / "summary.json"
        )

        try:
            file_size_bytes = await self._write_upload(
                upload=upload,
                destination=input_path,
            )
        except BaseException:
            self.delete_job_files(
                input_path=input_path,
                output_path=output_path,
            )
            raise
        finally:
            await upload.close()

        return StoredVideo(
            original_filename=original_filename,
            stored_filename=stored_filename,
            input_path=input_path,
            output_path=output_path,
            summary_path=summary_path,
            content_type=(
                upload.content_type
                or "application/octet-stream"
            ),
            file_size_bytes=file_size_bytes,
        )

    async def _write_upload(
        self,
        upload: UploadFile,
        destination: Path,
    ) -> int:
        total_bytes = 0

        with destination.open("wb") as destination_file:
            while True:
                chunk = await upload.read(
                    self._settings.upload_chunk_size_bytes
                )

                if not chunk:
                    break

                total_bytes += len(chunk)

                if (
                    total_bytes
                    > self._settings.maximum_upload_size_bytes
                ):
                    raise VideoUploadError(
                        code="VIDEO_TOO_LARGE",
                        message=(
                            "The uploaded video exceeds the "
                            f"{self._settings.maximum_upload_size_mb} "
                            "MB limit."
                        ),
                    )

                destination_file.write(chunk)

        if total_bytes == 0:
            raise VideoUploadError(
                code="EMPTY_VIDEO",
                message="The uploaded video is empty.",
            )

        return total_bytes

    def delete_job_files(
        self,
        input_path: Path,
        output_path: Path,
    ) -> None:
        input_directory = input_path.parent
        output_directory = output_path.parent

        for path in (
            input_directory,
            output_directory,
        ):
            if not path.exists():
                continue

            for child in path.iterdir():
                if child.is_file():
                    child.unlink()

            try:
                path.rmdir()
            except OSError:
                pass

    @staticmethod
    def _validate_filename(
        filename: str | None,
    ) -> str:
        if not filename:
            raise VideoUploadError(
                code="MISSING_FILENAME",
                message="The uploaded video has no filename.",
            )

        safe_filename = Path(filename).name

        if safe_filename in {"", ".", ".."}:
            raise VideoUploadError(
                code="INVALID_FILENAME",
                message="The uploaded filename is invalid.",
            )

        return safe_filename

    def _validate_extension(
        self,
        extension: str,
    ) -> None:
        if extension not in self._settings.allowed_video_extensions:
            allowed = ", ".join(
                self._settings.allowed_video_extensions
            )

            raise VideoUploadError(
                code="UNSUPPORTED_VIDEO_EXTENSION",
                message=(
                    "Unsupported video extension. "
                    f"Allowed extensions: {allowed}."
                ),
            )

    def _validate_content_type(
        self,
        content_type: str | None,
    ) -> None:
        if (
            content_type
            not in self._settings.allowed_video_content_types
        ):
            raise VideoUploadError(
                code="UNSUPPORTED_VIDEO_CONTENT_TYPE",
                message=(
                    "The uploaded file does not have a "
                    "supported MP4 content type."
                ),
            )