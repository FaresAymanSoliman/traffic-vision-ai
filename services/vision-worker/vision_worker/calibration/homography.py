import cv2 
import numpy as np 

from vision_worker.calibration.models import (
    CalibrationConfig ,
    WorldPoint ,
)

from vision_worker.tracking.models import Point

class HomographyTransformer:
    def __init__(self, config: CalibrationConfig) -> None: 
        self._config = config

        source_points = np.array(
            [
                [config.top_left.x, config.top_left.y],
                [config.top_right.x, config.top_right.y],
                [config.bottom_right.x, config.bottom_right.y],
                [config.bottom_left.x, config.bottom_left.y],
            ],
            dtype=np.float32
        )

        destination_points = np.array(
            [
                [0.0,0.0],
                [config.road_width_meters,0.0],
                [config.road_width_meters, config.road_length_meters],
                [0.0, config.road_length_meters],

            ],
            dtype = np.float32,
        )

        self._matrix = cv2.getPerspectiveTransform(source_points, destination_points)

        if not np.isfinite(self._matrix).all():
            raise ValueError ("The calibration produced an invalid homography matrix.")

    @property
    def config(self) -> CalibrationConfig:
        return self._config

    @property 
    def matrix (self) -> np.ndarray:
        return self._matrix.copy()


    def transform (self, point: Point) -> WorldPoint:
        source = np.array(
            [[[float(point.x), float(point.y)]]], dtype=np.float32
        )

        transformed = cv2.perspectiveTransform(source, self._matrix)

        x_meters = float(transformed[0][0][0])
        y_meters = float(transformed[0][0][1])

        if not np.isfinite([x_meters, y_meters]).all():
            raise ValueError("The homography transformation produced invalid coordinates.")

        return WorldPoint (
            x_meters = x_meters,
            y_meters = y_meters,
        )


    