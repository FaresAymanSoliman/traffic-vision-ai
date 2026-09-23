import json
import time
from collections.abc import Callable
from pathlib import Path

import cv2
from cv2.typing import MatLike

from vision_worker.video.models import ProcessingSummary, VideoMetadata

progressCallback = Callable[[int, int, float], None]


class VideoProcessingError(Exception):
    """Raised when a video cannot be read, processed, or written."""


class VideoProcessor:
    def __init__(self, codec: str = "mp4v") -> None:
        if len(codec) != 4:
            raise ValueError("The video codec must contain exactly four characters")
        self._codec = codec

    def read_metadata(self, input_path: Path) -> VideoMetadata:
        input_path = input_path.resolve()

        if not input_path.exists():
            raise VideoProcessingError(f"Input video file does not exist: {input_path}")

        if not input_path.is_file():
            raise VideoProcessingError(f"Input video path is not a file: {input_path}")

        capture = cv2.VideoCapture(str(input_path))

        try:
            if not capture.isOpened():
                raise VideoProcessingError(
                    f"OpenCv could not open the video file: {input_path}"
                )
            return self._extract_metadata(capture)

        finally:
            capture.release()

    def process(
        self,
        input_path: Path,
        output_path: Path,
        summary_path: Path | None = None,
        progress_callback: progressCallback | None = None,
    ) -> ProcessingSummary:
        input_path = input_path.resolve()
        output_path = output_path.resolve()

        if not input_path.exists():
            raise VideoProcessingError(f"Input video file does not exist: {input_path}")

        if not input_path.is_file():
            raise VideoProcessingError(f"Input video path is not a file: {input_path}")

        if input_path == output_path:
            raise VideoProcessingError(
                f"Input and output video paths cannot be the same: {input_path}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(input_path))

        if not capture.isOpened():
            capture.release()
            raise VideoProcessingError(
                f"OpenCv could not open the video file: {input_path}"
            )

        writer: cv2.VideoWrite | None = None
        processed_frames = 0
        started_at = time.perf_counter()

        try:
            metadata = self._extract_metadata(capture)
            writer = self._create_writer(output_path, metadata)

            while True:
                frame_read, frame = capture.read()
                if not frame_read:
                    break
                processed_frames += 1

                timestamp_seconds = (
                    (processed_frames - 1) / metadata.fps if metadata.fps > 0 else 0.0
                )

                processed_frame = self._process_frame(
                    frame=frame,
                    frame_number=processed_frames,
                    timestamp_seconds=timestamp_seconds,
                    metadata=metadata,
                )

                writer.write(processed_frame)

                if progress_callback is not None:
                    progress = self._calculate_progress(
                        processed_frames=processed_frames,
                        total_frames=metadata.frame_count,
                    )
                    progress_callback(
                        processed_frames,
                        metadata.frame_count,
                        progress,
                    )
                    
            if processed_frames == 0:
                raise VideoProcessingError(
                    f"No frames were processed from the video file: {input_path}"
                )
        except Exception:
            if output_path.exists():
                output_path.unlink()
            raise
        finally:
            capture.release()

            if writer is not None:
                writer.release()

        elapsed_seconds = time.perf_counter() - started_at
        average_processing_fps = (
            processed_frames / elapsed_seconds if elapsed_seconds > 0 else 0.0
        )

        summary = ProcessingSummary(
            input_path=input_path,
            output_path=output_path,
            metadata=metadata,
            processed_frames=processed_frames,
            processing_time_seconds=elapsed_seconds,
            average_processing_fps=average_processing_fps,
        )
        if summary_path is not None:
            self._write_summary(summary, summary_path.resolve())

        return summary

    def _extract_metadata(self, capture: cv2.VideoCapture) -> VideoMetadata:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            raise VideoProcessingError(
                "Invalid video dimensions: width and height must be positive integers."
            )

        if fps <= 0:
            raise VideoProcessingError(
                "Invalid video FPS: FPS must be a positive number."
            )

        return VideoMetadata(
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
        )

    def _create_writer(
        self,
        output_path: Path,
        metadata: VideoMetadata,
    ) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            metadata.fps,
            (metadata.width, metadata.height),
        )
        if not writer.isOpened():
            raise VideoProcessingError(
                f"OpenCv could not create the video writer for: {output_path}"
            )
        return writer

    def _process_frame(
        self,
        frame: MatLike,
        frame_number: int,
        timestamp_seconds: float,
        metadata: VideoMetadata,
    ) -> MatLike:
        overlay_height = min(100, max(70, metadata.height // 8))
        cv2.rectangle(
            frame,
            (0, 0),
            (metadata.width, overlay_height),
            color=(15, 23, 42),
            thickness=-1,
        )

        cv2.putText(
            frame,
            "TrafficVision AI",
            (24, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (85, 214, 190),
            2,
            cv2.LINE_AA,
        )

        information = (
            f"Frame: {frame_number} | "
            f"Time: {timestamp_seconds:.2f}s"
            f"Resolution: {metadata.width} x {metadata.height}"
        )

        cv2.putText(
            frame,
            information,
            (24, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (232, 240, 247),
            1,
            cv2.LINE_AA,
        )

        return frame

    @staticmethod
    def _calculate_progress(
        processed_frames: int,
        total_frames: int,
    ) -> float:
        if total_frames <= 0:
            return 0.0

        return min((processed_frames / total_frames) * 100, 100.0)

    @staticmethod
    def _write_summary(
        summary: ProcessingSummary,
        summary_path: Path,
    ) -> None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        summary_path.write_text(
            json.dumps(summary.to_dict(), indent=2),
            encoding="utf-8",
        )
