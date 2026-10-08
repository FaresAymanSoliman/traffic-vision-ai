import cv2 
import numpy as np 

from vision_worker.tracking.models import Point , TrackedVehicle

class RoadOccupancyCalculator:
    def __init__ (
            self,
            road_polygon: tuple[Point,...],
            frame_width: int,
            frame_height,
    ) -> None:
        if len(road_polygon) < 3:
            raise ValueError(
                "Road polygon requires at least three points."
            )

        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(
                "Frame dimensions must be greater then zero."
            )

        self._road_polygon = road_polygon
        self._frame_width = frame_width
        self._frame_height = frame_height


        self._road_mask = np.zeros(
            (frame_height, frame_width),
            dtype = np.uint8
        )

        polygon_array = np.array(
            [
                [point.x, point.y] for point in road_polygon
            ],
            dtype = np.int32,
        )

        cv2.fillPoly(
            self._road_mask,
            [polygon_array],
            color = 255,
        )

        self._road_area_pixels = int(
            cv2.countNonZero(self._road_mask)
        )

        if self._road_area_pixels <= 0:
            raise ValueError(
                "Road polygon has no measurable area."
            )

    @property
    def road_polygon(self) -> tuple[Point,...]:
        return self._road_polygon

    @property
    def road_area_pixels(self) -> int:
        return self._road_area_pixels

    def calculate(
            self,
            tracked_vehicles: list[TrackedVehicle],
    ) -> float:
        vehicle_mask = np.zeros(
            (self._frame_height,self._frame_width),
            dtype = np.uint8,
        )

        for vehicle in tracked_vehicles:
            box = vehicle.bounding_box

            x1 = max(
                0,
                min(box.x1, self._frame_width - 1),
            )

            y1 = max(
                0,
                min(box.y1, self._frame_height - 1),
            )

            x2 = max(
                0,
                min(box.x2, self._frame_width - 1),
            )

            y2 = max(
                0,
                min(box.y2, self._frame_height - 1)
            )

            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(
                vehicle_mask,
                (x1,y1),
                (x2,y2),
                color = 255,
                thickness = -1,
            )
        occupied_road_mask = cv2.bitwise_and(
            vehicle_mask,
            self._road_mask,
        )

        occupied_pixels = int(
            cv2.countNonZero(occupied_road_mask)
        )

        occupancy_ratio = (
            occupied_pixels / self._road_area_pixels
        )

        return min(
            max(occupancy_ratio, 0.0),
            1.0
        )
