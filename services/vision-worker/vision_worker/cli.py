import argparse
import json
import sys
from pathlib import Path

from vision_worker.detection.yolo_detector import (
    DetectorConfigurationError,
    YoloVehicleDetector,
)
from vision_worker.tracking.yolo_tracker import (
    TrackerConfigurationError,
    YoloVehicleTracker,
)
from vision_worker.video.processor import (
    VideoProcessingError,
    VideoProcessor,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Detect vehicles in a traffic video using YOLO.")
    )

    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker configuration,",
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input MP4 video.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the processed output MP4 video.",
    )

    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path for the JSON processing summary.",
    )

    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="YOLO model name or path.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum detection confidence between 0 and 1.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="YOLO inference image size.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, such as cpu, 0, or cuda:0.",
    )

    return parser.parse_args()


def display_progress(
    processed_frames: int,
    total_frames: int,
    progress: float,
) -> None:
    total_display = str(total_frames) if total_frames > 0 else "unknown"

    print(
        (f"\rProcessed {processed_frames}/{total_display} frames ({progress:.1f}%)"),
        end="",
        flush=True,
    )


def main() -> None:
    arguments = parse_arguments()

    try:
        detector = YoloVehicleDetector(
            model_path=arguments.model,
            confidence_threshold=arguments.confidence,
            image_size=arguments.image_size,
            device=arguments.device,
        )

        tracker = YoloVehicleTracker(
            model_path=arguments.model,
            tracker_config=arguments.tracker,
            confidence_threshold=arguments.confidence,
            image_size=arguments.image_size,
            device=arguments.device,
        )

        processor = VideoProcessor(detector=detector)
        processor = VideoProcessor(tracker=tracker)

        summary = processor.process(
            input_path=arguments.input,
            output_path=arguments.output,
            summary_path=arguments.summary,
            progress_callback=display_progress,
        )

        print()
        print("Vehicle detection complete.")
        print("Vehicle tracking complete.")
        print(json.dumps(summary.to_dict(), indent=2))

    except (
        DetectorConfigurationError,
        VideoProcessingError,
        TrackerConfigurationError,
    ) as error:
        print(
            f"\nVehicle detection failed: {error}",
            file=sys.stderr,
        )

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
