from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "TrafficVision AI API"
    environment: str = "development"
    debug: bool = True
    api_prefix: str = "/api"

    frontend_origin: str = "http://localhost:5173"

    project_root: Path = Path(__file__).resolve().parents[4]

    data_directory: Path | None = None

    maximum_upload_size_mb: int = 250
    upload_chunk_size_bytes: int = 1024 * 1024

    allowed_video_extensions: tuple[str, ...] = (
        ".mp4",
    )

    allowed_video_content_types: tuple[str, ...] = (
        "video/mp4",
        "application/mp4",
        "application/octet-stream",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def resolved_data_directory(self) -> Path:
        if self.data_directory is not None:
            return self.data_directory.resolve()

        return (self.project_root / "data").resolve()

    @property
    def input_jobs_directory(self) -> Path:
        return self.resolved_data_directory / "input" / "jobs"

    @property
    def output_jobs_directory(self) -> Path:
        return self.resolved_data_directory / "output" / "jobs"

    @property
    def maximum_upload_size_bytes(self) -> int:
        return self.maximum_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()