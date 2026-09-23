from dataclasses import asdict, dataclass, field
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
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "duration_seconds": round(self.duration_seconds, 3),
        }


@dataclass(frozen=True, slots=True)
class DetectionStatistics:
    total_detections: int = 0
    maximum_vehicles_in_frame: int = 0
    class_detection_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "total_detections": self.total_detections,
            "maximum_vehicles_in_frame": self.maximum_vehicles_in_frame,
            "class_detection_counts": self.class_detection_counts,
        }


@dataclass(frozen=True, slots=True)
class ProcessingSummary:
    input_path: Path
    output_path: Path
    metadata: VideoMetadata
    processed_frames: int
    processing_time_seconds: float
    average_processing_fps: float
    model_name: str | None = None
    detection_statistics: DetectionStatistics | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["input_path"] = str(self.input_path)
        result["output_path"] = str(self.output_path)
        result["metadata"] = self.metadata.to_dict()

        result["processing_time_seconds"] = round(
            self.processing_time_seconds,
            3,
        )

        result["average_processing_fps"] = round(
            self.average_processing_fps,
            3,
        )

        if self.detection_statistics is not None:
            result["detection_statistics"] = self.detection_statistics.to_dict()

        return result
