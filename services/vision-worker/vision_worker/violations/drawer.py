import cv2
from cv2.typing import MatLike

from vision_worker.tracking.models import TrackedVehicle
from vision_worker.violations.models import ViolationEvent

VIOLATION_COLOR = (60, 60, 255)


def draw_wrong_way_information(
    frame: MatLike,
    tracked_vehicles: list[TrackedVehicle],
    reported_track_ids: set[int],
    new_events: list[ViolationEvent],
    total_events: int,
) -> MatLike:
    for vehicle in tracked_vehicles:
        if vehicle.track_id not in reported_track_ids:
            continue

        box = vehicle.bounding_box

        cv2.rectangle(
            frame,
            (box.x1, box.y1),
            (box.x2, box.y2),
            VIOLATION_COLOR,
            thickness=4,
        )

        label_y = max(box.y1 - 12, 25)

        cv2.putText(
            frame,
            "WRONG WAY",
            (box.x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            VIOLATION_COLOR,
            2,
            cv2.LINE_AA,
        )

    for event in new_events:
        cv2.circle(
            frame,
            (
                event.reference_point.x,
                event.reference_point.y,
            ),
            radius=14,
            color=VIOLATION_COLOR,
            thickness=3,
        )

    panel_width = 285
    panel_height = 50
    panel_margin = 20

    panel_x = max(
        frame.shape[1] - panel_width - panel_margin,
        0,
    )
    panel_y = 250

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

    cv2.putText(
        frame,
        f"Wrong-way events: {total_events}",
        (panel_x + 14, panel_y + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        VIOLATION_COLOR,
        2,
        cv2.LINE_AA,
    )

    return frame