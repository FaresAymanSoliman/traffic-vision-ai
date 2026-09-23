import cv2
from cv2.typing import MatLike

from vision_worker.detection.models import Detection

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (85, 214, 190),
    "motorcycle": (244, 114, 182),
    "bus": (251, 191, 36),
    "truck": (96, 165, 250),
}

DEFAULT_COLOR = (203, 213, 225)


def draw_detections(
    frame: MatLike,
    detections: list[Detection],
) -> MatLike:
    for detection in detections:
        box = detection.bounding_box

        color = CLASS_COLORS.get(
            detection.class_name,
            DEFAULT_COLOR,
        )

        cv2.rectangle(
            frame,
            (box.x1, box.y1),
            (box.x2, box.y2),
            color,
            thickness=2,
        )

        label = f"{detection.class_name} {detection.confidence:.2f}"

        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )

        text_width, text_height = text_size

        label_top = max(box.y1 - text_height - baseline - 10, 0)
        label_bottom = label_top + text_height + baseline + 10

        cv2.rectangle(
            frame,
            (box.x1, label_top),
            (box.x1 + text_width + 12, label_bottom),
            color,
            thickness=-1,
        )

        cv2.putText(
            frame,
            label,
            (box.x1 + 6, label_bottom - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (15, 23, 42),
            2,
            cv2.LINE_AA,
        )

    return frame
