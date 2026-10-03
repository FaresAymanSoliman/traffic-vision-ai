import cv2 
import numpy as np 
from cv2.typing import MatLike

from vision_worker.calibration.models import CalibrationConfig

CALIBRATION_COLOR = (190, 120, 255)


def draw_calibration_region(
        frame:  MatLike,
        config: CalibrationConfig,
) -> MatLike:
    polygon = np.array(
        [
            [point.x , point.y] for point in config.image_points
        ],
        dtype = np.int32
    ).reshape((-1, 1, 2))

    cv2.polylines(
        frame,
        [polygon],
        isClosed = True,
        color = CALIBRATION_COLOR,
        thickness = 2,
        lineType = cv2.LINE_AA
    )

    labels = ("TL", "TR", "BR", "BL")

    for label, point in zip(
        labels,
        config.image_points,
        strict = True,
    ):
        cv2.circle(
            frame,
            (point.x, point.y),
            radius = 6,
            color = CALIBRATION_COLOR,
            thickness = -1,
        )

        cv2.putText(
            frame,
            label,
            (point.x + 8, point.y -8),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.55,
            color = CALIBRATION_COLOR,
            thickness = 1,
            lineType = cv2.LINE_AA
        )

    return frame
