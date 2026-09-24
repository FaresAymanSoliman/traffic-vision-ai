from pathlib import Path
from typing import Any

from cv2.typing import MatLike
from ultralytics import YOLO

from vision_worker.detection.models import BoundingBox
from vision_worker.tracking.models import TrackedVehicle

VEHICLE_CLASS_NAMES = frozenset(
    {
        "car",
        "motorcycle",
        "bus",
        "truck",
    }
)


class TrackerConfigurationError(ValueError):
    """Raised when vehicle-tracker configuration is invalid"""


class YoloVehicleTracker:
    def __init__(
        self,
        model_path: str | Path = "yolo11n.pt",
        tracker_config: str = "bytetrack.yaml",
        confidence_threshold: float = 0.35,
        image_size: int = 640,
        device: str | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise TrackerConfigurationError(
                "Confidence threshold must be between 0 and 1"
            )

        if image_size <= 0:
            raise TrackerConfigurationError("Image size must be greater than 0.")

        self._model_path = str(model_path)
        self._tracker_config = tracker_config
        self._confidence_threshold = confidence_threshold
        self._image_size = image_size
        self._device = device

        print(f"Loading YOLO model: {self._model_path}")
        print(f"Tracker configuration: {self._tracker_config}")

        self._model = YOLO(self._model_path)
        self._supported_class_ids = self._find_supported_class_ids()

        if not self._supported_class_ids:
            raise TrackerConfigurationError(
                "The selected model does not contain supported vehicle classes"
            )

        print("supported vehicle classes: ", self.supported_class_names)

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def tracker_config(self) -> str:
        self._tracker_config

    @property
    def supported_class_names(self) -> list:
        return sorted(
            str(self._model.names[class_id]) for class_id in self._supported_class_ids
        )

    def track(self, frame: MatLike) -> list:
        tracking_arguments: dict[str, Any] = {
            "source": frame,
            "persist": True,
            "tracker": self._tracker_config,
            "conf": self._confidence_threshold,
            "imgsz": self._image_size,
            "classes": sorted(self._supported_class_ids),
            "verbose": False,
        }

        if self._device is not None:
            tracking_arguments["device"] = self._device

        results = self._model.track(**tracking_arguments)

        if not results:
            return []

        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            return []

        tracked_vehicles: list[TrackedVehicle] = []

        for box in result.boxes:
            if box.id is None:
                continue

            track_id = int(box.id.item())
            class_id = int(box.cls.item())
            class_name = str(self._model.names[class_id])
            confidence = float(box.conf.item())

            coordinates = box.xyxy[0].cpu().tolist()

            x1, y1, x2, y2 = (int(round(coordinate)) for coordinate in coordinates)

            tracked_vehicles.append(
                TrackedVehicle(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bounding_box=BoundingBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    ),
                )
            )

        return tracked_vehicles

    def _find_supported_class_ids(self) -> set:
        return {
            int(class_id)
            for class_id, class_name in self._model.names.items()
            if str(class_name) in VEHICLE_CLASS_NAMES
        }
