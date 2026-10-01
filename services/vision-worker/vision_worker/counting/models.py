from dataclasses import dataclass
from enum import Enum 

from vision_worker.tracking.models import Point


class CrossingDirection(str,Enum):
    FORWARD = "forward"
    REVERSE = "reverse"


@dataclass(frozen = True , slots = True)
class CountingLine:
    line_id: str
    start: Point
    end: Point

    def __post_init__(self) -> None:
        if not self.line_id.strip():
            raise ValueError("Counting line ID cannot be empty")

        if self.start == self.end:
            raise ValueError(
                "Counting line start and end points must be different"
            )


    def to_dict(self) -> dict[str,object]:
        return{
            "line_id": self.line_id,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),

        }


@dataclass(frozen = True , slots = True)
class CountEvent:
    track_id: int
    class_name: str
    line_id: str
    direction: CrossingDirection
    frame_number: int
    timestamp_seconds: float
    crossing_point: Point


    def to_dict(self) -> dict[str , object]:
        return{
            "track_id": self.track_id,
            "class_name": self.class_name,
            "line_id": self.line_id,
            "direction": self.direction.value,
            "frame_number": self.frame_number,
            "timestamp_seconds": round(self.timestamp_seconds,3),
            "crossing_point": self.crossing_point.to_dict(),
        }


@dataclass(frozen = True, slots= True)
class CountingSnapshot:
    total_crossings: int
    counts_by_class: dict[str,int]
    counts_by_direction: dict[str,int]

    def to_dict(self) -> dict[str , object]:
        return{
            "total_crossing": self.total_crossings,
            "counts_by_class": self.counts_by_class,
            "counts_by_direction": self.counts_by_direction,
        }

    