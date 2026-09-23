from pathlib import Path
from cv2.typing import MatLike
from ultralytics import YOLO
from vision_worker.detection.models import BoundingBox, Detection





VEHICLE_CLASS_NAMES = frozenset({"car", "motorcycle", "bus", "truck"})


class DetectorConfigurationError(Exception):
    """Raised when detector configuration is invalid."""


class YoloVehicleDetector:
    def __init__(
        self,
        model_path: str | Path = "yolo11n.pt",
        confidence_threshold: float = 0.35,
        image_size: int = 640,
        device: str | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise DetectorConfigurationError(
                "Confidence threshold must be between 0 and 1"
            )

        if image_size <= 0:
            raise DetectorConfigurationError("Image size must be grater than 0")

        self._model_path = str(model_path)
        self._confidence_threshold = confidence_threshold
        self._image_size = image_size
        self._device = device

        print(f"Loading YOLO model: {self._model_path}")

        self._model = YOLO(self._model_path)
        self._supported_class_ids = self._find_supported_class_ids()

        if not self._supported_class_ids:
            raise DetectorConfigurationError(
                "The selected model does not contain supported vehicle classes."
            )

        print(
            "supported vehicle classes: ",
            self.supported_class_names,
        )

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @property
    def supported_class_names(self) -> list[str]:
        names = self._model.names
        return sorted(str(names[class_id]) for class_id in self._supported_class_ids)

    def detect(self, frame: MatLike) -> list:
        prediction_arguments = {
            "source": frame,
            "conf": self._confidence_threshold,
            "imgsz": self._image_size,
            "classes": sorted(self._supported_class_ids),
            "verbose": False,
        }

        if self._device is not None:
            prediction_arguments["device"] = self._device

        results = self._model.predict(**prediction_arguments)

        if not results:
            return []
        result = results[0]

        if result.boxes is None:
            return []

        detections : list[Detection] = []

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            coordinates = box.xyxy[0].cpu().tolist()

            x1, y1, x2, y2 = (int(round(coordinate)) for coordinate in coordinates)

            class_name = str(self._model.names[class_id])

            detections.append(
                Detection(
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

        return detections

    def _find_supported_class_ids(self) -> set[int]:
        return {
            int(class_id)
            for class_id, class_name in self._model.names.items()
            if str(class_name) in VEHICLE_CLASS_NAMES
        }
