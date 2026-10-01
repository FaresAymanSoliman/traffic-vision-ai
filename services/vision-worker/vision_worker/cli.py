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

from vision_worker.counting.line_counter import LineCounter
from vision_worker.counting.models import CountingLine
from vision_worker.tracking.models import Point




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

    parser.add_argument(
        "--line-id",
        default = "main-road",
        help = "Identifier for the counting line.",
    )

    parser.add_argument(
        "--line-start-x",
        type = int,
        required = True,
        help = "Counting-line start x coordinate.",
    )

    parser.add_argument(
        "--line-start-y",
        type = int,
        required = True,
        help = "Counting-line start y coordinate.",
    )

    parser.add_argument(
        "--line-end-x",
        type = int,
        required = True,
        help = "Counting-line end x coordinate.",
    )

    parser.add_argument(
        "--line-end-y",
        type = int,
        required = True,
        help = "Counting-line end y coordinate.",
    )

    parser.add_argument(
        "--minimum-movement",
        type = float,
        default = 2.0,
        help = "Minimum vehicle movement (in pixels) to count as a crossing event.",
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

        counting_line = CountingLine(
            line_id = arguments.line_id,
            start = Point(
                x = arguments.line_start_x,
                y = arguments.line_start_y,

            ),

            end = Point(
                x = arguments.line_end_x,
                y = arguments.line_end_y,
            ),
        )

        line_counter = LineCounter(
            counting_line = counting_line,
            minimum_movement_pixels = arguments.minimum_movement,
        )

        #processor = VideoProcessor(detector=detector)
        processor = VideoProcessor(tracker=tracker, line_counter=line_counter)


        summary = processor.process(
            input_path=arguments.input,
            output_path=arguments.output,
            summary_path=arguments.summary,
            progress_callback=display_progress,
        )

        print()
        print("Vehicle detection complete.")
        print("Vehicle tracking and counting complete.")
        print(json.dumps(summary.to_dict(), indent=2))

    except (
        DetectorConfigurationError,
        VideoProcessingError,
        TrackerConfigurationError,
        ValueError,
    ) as error:
        print(
            f"\nVehicle detection failed: {error}",
            file=sys.stderr,
        )

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
