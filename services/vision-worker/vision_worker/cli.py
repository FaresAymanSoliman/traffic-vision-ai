import argparse
import json
import sys
from pathlib import Path
import cv2

from vision_worker.density.estimator import DensityEstimator
from vision_worker.density.models import DensityConfig
from vision_worker.density.occupancy import (
    RoadOccupancyCalculator,
)

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

from vision_worker.calibration.homography import (
    HomographyTransformer,
)
from vision_worker.calibration.models import CalibrationConfig
from vision_worker.speed.estimator import SpeedEstimator





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

    parser.add_argument("--calibration-tl-x", type = int, required = True)
    parser.add_argument("--calibration-tl-y", type = int, required = True)

    parser.add_argument("--calibration-tr-x", type = int, required = True)
    parser.add_argument("--calibration-tr-y", type = int, required = True)

    parser.add_argument("--calibration-br-x", type = int, required = True)
    parser.add_argument("--calibration-br-y", type = int, required = True)

    parser.add_argument("--calibration-bl-x", type = int, required = True)
    parser.add_argument("--calibration-bl-y", type = int, required = True)


    parser.add_argument(
        "--road-width-meters",
        type = float,
        required = True,
    )

    parser.add_argument(
        "--road-length-meters",
        type = float,
        required = True,
    )

    parser.add_argument(
        "--speed-history-seconds",
        type = float,
        default = 1.0,
    )

    parser.add_argument(
        "--minimum-speed-observation-seconds",
        type = float,
        default = 0.4,
    )

    parser.add_argument(
        "--maximum-speed-kmh",
        type = float,
        default = 180.0,

    )

    parser.add_argument(
        "--medium-density-vehicles",
        type = int,
        default=7,
    )

    parser.add_argument(
        "--high-density-vehicles",
        type = int,
        default = 15,
    )

    parser.add_argument(
        "--medium-occupancy-ratio",
        type = float,
        default = 0.18,
    )

    parser.add_argument(
        "--high-occupancy-ratio",
        type = float,
        default = 0.40,
    )

    parser.add_argument(
        "--density-low-speed-kmh",
        type = float,
        default = 20.0,
    )

    parser.add_argument(
        "--density-minimum-speed-rule-vehicles",
        type =int,
        default = 5,
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

def read_video_dimensions(
    input_path: Path,
) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(input_path))

    try:
        if not capture.isOpened():
            raise ValueError(
                f"Could not open input video: {input_path}"
            )

        width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        if width <= 0 or height <= 0:
            raise ValueError(
                "Input video has invalid dimensions."
            )

        return width, height
    finally:
        capture.release()


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

        calibration_config = CalibrationConfig(
            top_left =Point(
                x = arguments.calibration_tl_x,
                y = arguments.calibration_tl_y,
            ),
            top_right = Point(
                x = arguments.calibration_tr_x,
                y = arguments.calibration_tr_y,
            ),

            bottom_right = Point(
                x = arguments.calibration_br_x, 
                y = arguments.calibration_br_y,
            ),

            bottom_left = Point(
                x = arguments.calibration_bl_x,
                y = arguments.calibration_bl_y,
            ),

            road_width_meters = arguments.road_width_meters,
            road_length_meters = arguments.road_length_meters,

        )

        frame_width, frame_height = read_video_dimensions(
            arguments.input
        )

        road_polygon = (
            calibration_config.top_left,
            calibration_config.top_right,
            calibration_config.bottom_right,
            calibration_config.bottom_left,
        )

        occupancy_calculator = RoadOccupancyCalculator(
            road_polygon=road_polygon,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        density_config = DensityConfig(
            medium_vehicle_count=(
                arguments.medium_density_vehicles
            ),
            high_vehicle_count=(
                arguments.high_density_vehicles
            ),
            medium_occupancy_ratio=(
                arguments.medium_occupancy_ratio
            ),
            high_occupancy_ratio=(
                arguments.high_occupancy_ratio
            ),
            low_speed_threshold_kmh=(
                arguments.density_low_speed_kmh
            ),
            minimum_vehicles_for_speed_rule=(
                arguments.density_minimum_speed_rule_vehicles
            ),
        )

        density_estimator = DensityEstimator(
            occupancy_calculator=occupancy_calculator,
            config=density_config
        )

        transformer = HomographyTransformer(calibration_config)

        speed_estimator = SpeedEstimator(
            transformer=transformer,
            history_seconds=arguments.speed_history_seconds,
            minimum_observation_seconds=(arguments.minimum_speed_observation_seconds),
            maximum_speed_kmh=arguments.maximum_speed_kmh,

        )

        #processor = VideoProcessor(detector=detector)
        processor = VideoProcessor(tracker=tracker, line_counter=line_counter, speed_estimator=speed_estimator,density_estimator=density_estimator,)


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
