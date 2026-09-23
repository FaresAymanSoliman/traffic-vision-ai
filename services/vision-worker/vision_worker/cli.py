import argparse
import json
import sys
from pathlib import Path

from vision_worker.video.processor import (
    VideoProcessingError,
    VideoProcessor,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Process a traffic video and create an annotated output video.")
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
    processor = VideoProcessor()

    try:
        summary = processor.process(
            input_path=arguments.input,
            output_path=arguments.output,
            summary_path=arguments.summary,
            progress_callback=display_progress,
        )

        print()
        print("Video processing complete.")
        print(json.dumps(summary.to_dict(), indent=2))
    except VideoProcessingError as error:
        print(f"\nVideo processing failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
