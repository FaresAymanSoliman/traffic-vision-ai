from dataclasses import dataclass 

from vision_worker.tracking.models import Point 


@dataclass (frozen = True , slots = True )
class WorldPoint:
    x_meters: float 
    y_meters: float 

    def to_dict(self) -> dict[str , float]:
        return {
            "x_meters": round(self.x_meters, 3),
            "y_meters": round(self.y_meters, 3),
        }



@dataclass (frozen = True , slots = True )
class CalibrationConfig:
    top_left: Point
    top_right: Point
    bottom_right: Point 
    bottom_left: Point
    road_width_meters: float
    road_length_meters: float

    def __post_init__(self) -> None:
        if self.road_width_meters <= 0:
            raise ValueError("road_width_meters must be greater than 0")
        if self.road_length_meters <= 0:
            raise ValueError("road_length_meters must be greater than 0")

        points ={
            self.top_left,
            self.top_right,
            self.bottom_left,
            self.bottom_right,
        }

        if len(points) != 4:
            raise ValueError(
                "Calibration requires four different image points."
            )

    @property 
    def image_points(self) -> tuple[Point , Point , Point , Point]:
        return (
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.bottom_left,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "top_left": self.top_left.to_dict(),
            "top_right": self.top_right.to_dict(),
            "bottom_right": self.bottom_right.to_dict(),
            "bottom_left": self.bottom_left.to_dict(),
            "road_width_meters": self.road_width_meters, 
            "road_length_meters": self.road_length_meters,
        }
    
