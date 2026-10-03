import cv2
from cv2.typing import MatLike

from vision_worker.speed.models import VehicleSpeed
from vision_worker.tracking.models import TrackedVehicle

SPEED_COLOR = (70, 220, 255)


def draw_vehicle_speeds(
    frame: MatLike,
    tracked_vehicles: list[TrackedVehicle],
    speeds: dict[int, VehicleSpeed],
) -> MatLike:
    for vehicle in tracked_vehicles:
        measurement = speeds.get(vehicle.track_id)

        if measurement is None:
            continue

        box = vehicle.bounding_box

        label = (
            f"Estimated: "
            f"{measurement.estimated_speed_kmh:.1f} km/h"
        )

        label_y = min(
            box.y2 + 24,
            frame.shape[0] - 10,
        )

        cv2.putText(
            frame,
            label,
            (box.x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            SPEED_COLOR,
            2,
            cv2.LINE_AA,
        )

    return frame