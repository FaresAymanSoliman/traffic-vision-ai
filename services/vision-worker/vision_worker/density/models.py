from dataclasses import dataclass 
from enum import Enum 

class DensityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"



@dataclass (frozen = True, slots = True)
class DensityConfig:
    medium_vehicle_count: int = 7
    high_vehicle_count: int = 15 
    medium_occupancy_ratio: float = 0.18
    high_occupancy_ratio: float = 0.40
    low_speed_threshold_kmh: float = 20.0
    minimum_vehicles_for_speed_rule: int = 5

    def __post_init__(self) -> None:
        if self.medium_vehicle_count < 0:
            raise ValueError(
                "Medium vehicle cannot be negative."
            )

        if self.high_vehicle_count <= self.medium_vehicle_count:
            raise ValueError(
                "High vehicle count must exceed medium vehicle count."
            )

        if not 0.0 <= self.medium_occupancy_ratio <= 1.0:
            raise ValueError(
                "Medium occupancy ratio must be between 0 and 1."
            )

        if not 0.0 <= self.high_occupancy_ratio <= 1.0:
            raise ValueError(
                "High occupancy ratio must be between 0 and 1."
            )

        if (self.high_occupancy_ratio <= self.medium_occupancy_ratio):
            raise ValueError(
                "High occupancy ratio must exceed medium occupancy ratio."
            )

        if self.low_speed_threshold_kmh < 0:
            raise ValueError(
                "Low-speed threshold cannot be negative."
            )

        if self.minimum_vehicles_for_speed_rule < 1:
            raise ValueError(
                "Minimum vehicles foe the speed rule must be at least one."
            )


@dataclass (frozen = True , slots = True)
class DensityMeasurement:
    level: DensityLevel
    active_vehicle_count: int 
    occupancy_ratio: float
    average_estimated_speed_kmh: float | None

    def to_dict(self) -> dict[str, object]:
        return{
            "level": self.level.value,
            "active_vehicle_count": self.active_vehicle_count,
            "occupancy_ratio": round(self.occupancy_ratio,4),
            "average_estimated_speed_kmh": round(self.average_estimated_speed_kmh,2) if self.average_estimated_speed_kmh is not None else None
        }


    