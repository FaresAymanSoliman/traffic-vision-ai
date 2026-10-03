from dataclasses import dataclass 

from vision_worker.calibration.models import WorldPoint

@dataclass(frozen = True , slots = True)
class TimedWorldPosition:
    timestamp_seconds: float 
    position: WorldPoint


@dataclass(frozen = True , slots = True)
class VehicleSpeed:
    track_id: int
    class_name: str
    estimated_speed_kmh: float
    observation_seconds: float
    distance_meters: float

    def to_dict(self) -> dict[str, int | str | float]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "estimated_speed_kmh": round(self.estimated_speed_kmh,2),
            "observation_seconds": round(self.observation_seconds, 3),
            "distance_meters": round(self.distance_meters, 3),
        }

    