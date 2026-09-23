from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps

    def to_dict(self) -> dict[str, int | float]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True, slots=True)
class ProcessingSummary:
    input_path: Path
    output_path: Path
    metadata: VideoMetadata
    processed_frames: int
    processing_time_seconds: float
    average_processing_fps: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["input_path"] = str(self.input_path)
        result["output_path"] = str(self.output_path)
        result["metadata"] = self.metadata.to_dict()
        result["processing_time_seconds"] = round(self.processing_time_seconds, 3)
        result["average_processing_fps"] = round(self.average_processing_fps, 3)
        return result
