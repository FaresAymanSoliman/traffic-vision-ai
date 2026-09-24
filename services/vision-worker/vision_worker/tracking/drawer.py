import cv2
import numpy as np
from cv2.typing import MatLike

from vision_worker.tracking.history import TrackHistory
from vision_worker.tracking.models import TrackedVehicle

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (85, 214, 190),
    "motorcycle": (244, 114, 182),
    "bus": (251, 191, 36),
    "truck": (96, 165, 250),
}

DEFAULT_COLOR = (203, 213, 225)


def draw_tracks(
    frame: MatLike,
    tracked_vehicles: list[TrackedVehicle],
    track_history: TrackHistory,
) -> MatLike:
    for vehicle in tracked_vehicles:
        box = vehicle.bounding_box

        color = CLASS_COLORS.get(
            vehicle.class_name,
            DEFAULT_COLOR,
        )

        cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), color, thickness=2)

        label = f"{vehicle.class_name}ID: {vehicle.track_id}{vehicle.confidence: 2f}"

        _draw_label(
            frame=frame,
            label=label,
            x=box.x1,
            y=box.y1,
            color=color,
        )

        reference_point = vehicle.reference_point

        cv2.circle(
            frame,
            (reference_point.x, reference_point.y),
            radius=5,
            color=color,
            thickness=-1,
        )

        history_points = track_history.get_points(vehicle.track_id)

        if len(history_points) >= 2:
            polyline_points = np.array(
                [[point.x, point.y] for point in history_points],
                dtype=np.int32,
            ).reshape((-1, 1, 2))

            cv2.polylines(
                frame,
                [polyline_points],
                isClosed=False,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    return frame


def _draw_label(
    frame: MatLike,
    label: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    text_size, baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2,
    )

    text_width, text_height = text_size

    label_top = max(
        y - text_height - baseline - 10,
        0,
    )

    label_bottom = label_top + text_height + baseline + 10

    cv2.rectangle(
        frame, (x, label_top), (x + text_width + 12, label_bottom), color, thickness=-1
    )

    cv2.putText(
        frame,
        label,
        (x + 6, label_bottom - baseline - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (15, 23, 42),
        2,
        cv2.LINE_AA,
    )
