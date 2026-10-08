import cv2
import numpy as np
from cv2.typing import MatLike

from vision_worker.density.models import (
    DensityLevel,
    DensityMeasurement,
)
from vision_worker.tracking.models import Point

DENSITY_COLORS: dict[
    DensityLevel,
    tuple[int, int, int],
] = {
    DensityLevel.LOW: (85, 214, 190),
    DensityLevel.MEDIUM: (0, 165, 255),
    DensityLevel.HIGH: (80, 80, 255),
}

ROAD_POLYGON_COLOR = (170, 110, 255)


def draw_density_information(
    frame: MatLike,
    road_polygon: tuple[Point, ...],
    measurement: DensityMeasurement,
) -> MatLike:
    polygon_array = np.array(
        [
            [point.x, point.y]
            for point in road_polygon
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    cv2.polylines(
        frame,
        [polygon_array],
        isClosed=True,
        color=ROAD_POLYGON_COLOR,
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    color = DENSITY_COLORS[measurement.level]

    panel_width = 375
    panel_height = 120
    panel_margin = 20

    panel_x = max(
        frame.shape[1] - panel_width - panel_margin,
        0,
    )
    panel_y = 115

    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (
            panel_x + panel_width,
            panel_y + panel_height,
        ),
        (15, 23, 42),
        thickness=-1,
    )

    speed_text = (
        f"{measurement.average_estimated_speed_kmh:.1f} km/h"
        if measurement.average_estimated_speed_kmh is not None
        else "Unavailable"
    )

    lines = [
        (
            "Traffic density: "
            f"{measurement.level.value.upper()}"
        ),
        (
            "Active vehicles: "
            f"{measurement.active_vehicle_count}"
        ),
        (
            "Road occupancy: "
            f"{measurement.occupancy_ratio * 100:.1f}%"
        ),
        f"Average estimated speed: {speed_text}",
    ]

    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (
                panel_x + 14,
                panel_y + 27 + index * 26,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color if index == 0 else (232, 240, 247),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )

    return frame