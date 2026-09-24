from dataclasses import dataclass

from vision_worker.detection.models import BoundingBox


@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True, slots=True)
class TrackedVehicle:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox

    @property
    def center(self) -> Point:
        center_x, center_y = self.bounding_box.center

        return Point(
            x=center_x,
            y=center_y,
        )

    @property
    def reference_point(self) -> Point:
        return Point(
            x=self.bounding_box.x1 + self.bounding_box.width // 2,
            y=self.bounding_box.y2,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bounding_box": self.bounding_box.to_dict(),
            "center": self.center.to_dict(),
            "reference_point": self.reference_point.to_dict(),
        }
