from dataclasses import dataclass
from enum import Enum

from vision_worker.tracking.models import Point


class ViolationType(str, Enum):
    WRONG_WAY = "wrong_way"


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"


@dataclass(frozen=True, slots=True)
class WrongWayConfig:
    allowed_direction_x: float
    allowed_direction_y: float
    minimum_displacement_pixels: float = 20.0
    confirmation_observations: int = 5
    similarity_threshold: float = -0.5
    history_size: int = 10

    def __post_init__(self) -> None:
        direction_length_squared = (
            self.allowed_direction_x**2
            + self.allowed_direction_y**2
        )

        if direction_length_squared == 0:
            raise ValueError(
                "Allowed direction vector cannot be zero."
            )

        if self.minimum_displacement_pixels <= 0:
            raise ValueError(
                "Minimum displacement must be greater than zero."
            )

        if self.confirmation_observations <= 0:
            raise ValueError(
                "Confirmation observations must be greater than zero."
            )

        if not -1.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                "Similarity threshold must be between -1 and 1."
            )

        if self.history_size < 2:
            raise ValueError(
                "History size must contain at least two points."
            )


@dataclass(frozen=True, slots=True)
class ViolationEvent:
    event_id: str
    violation_type: ViolationType
    track_id: int
    class_name: str
    frame_number: int
    timestamp_seconds: float
    reference_point: Point
    direction_similarity: float
    confidence: float
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "violation_type": self.violation_type.value,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "frame_number": self.frame_number,
            "timestamp_seconds": round(
                self.timestamp_seconds,
                3,
            ),
            "reference_point": self.reference_point.to_dict(),
            "direction_similarity": round(
                self.direction_similarity,
                4,
            ),
            "confidence": round(self.confidence, 4),
            "review_status": self.review_status.value,
        }