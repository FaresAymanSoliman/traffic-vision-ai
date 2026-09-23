import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import cv2
from cv2.typing import MatLike

from vision_worker.detection.drawer import draw_detections
from vision_worker.detection.models import Detection
from vision_worker.detection.yolo_detector import YoloVehicleDetector
from vision_worker.video.models import (
    DetectionStatistics,
    ProcessingSummary,
    VideoMetadata,
)

ProgressCallback = Callable[[int, int, float], None]


class VideoProcessingError(Exception):
    """Raised when a video cannot be read, processed, or written."""


class VideoProcessor:
    def __init__(
        self,
        detector: YoloVehicleDetector | None = None,
        codec: str = "mp4v",
    ) -> None:
        if len(codec) != 4:
            raise ValueError("The video codec must contain exactly four characters.")

        self._detector = detector
        self._codec = codec

    def read_metadata(self, input_path: Path) -> VideoMetadata:
        input_path = input_path.resolve()

        if not input_path.exists():
            raise VideoProcessingError(f"Input video does not exist: {input_path}")

        if not input_path.is_file():
            raise VideoProcessingError(f"Input path is not a file: {input_path}")

        capture = cv2.VideoCapture(str(input_path))

        try:
            if not capture.isOpened():
                raise VideoProcessingError(
                    f"OpenCV could not open the input video: {input_path}"
                )

            return self._extract_metadata(capture)
        finally:
            capture.release()

    def process(
        self,
        input_path: Path,
        output_path: Path,
        summary_path: Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ProcessingSummary:
        input_path = input_path.resolve()
        output_path = output_path.resolve()

        if not input_path.exists():
            raise VideoProcessingError(f"Input video does not exist: {input_path}")

        if not input_path.is_file():
            raise VideoProcessingError(f"Input path is not a file: {input_path}")

        if input_path == output_path:
            raise VideoProcessingError("Input and output paths must be different.")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(input_path))

        if not capture.isOpened():
            capture.release()
            raise VideoProcessingError(
                f"OpenCV could not open the input video: {input_path}"
            )

        writer: cv2.VideoWriter | None = None
        processed_frames = 0
        total_detections = 0
        maximum_vehicles_in_frame = 0
        class_detection_counts: Counter[str] = Counter()

        started_at = time.perf_counter()

        try:
            metadata = self._extract_metadata(capture)
            writer = self._create_writer(output_path, metadata)

            while True:
                frame_read, frame = capture.read()

                if not frame_read:
                    break

                if frame is None or frame.size == 0:
                    raise VideoProcessingError("OpenCV returned an empty video frame.")

                processed_frames += 1

                timestamp_seconds = (
                    (processed_frames - 1) / metadata.fps if metadata.fps > 0 else 0.0
                )

                detections = self._detect_vehicles(frame)

                total_detections += len(detections)

                maximum_vehicles_in_frame = max(
                    maximum_vehicles_in_frame,
                    len(detections),
                )

                class_detection_counts.update(
                    detection.class_name for detection in detections
                )

                processed_frame = self._process_frame(
                    frame=frame,
                    frame_number=processed_frames,
                    timestamp_seconds=timestamp_seconds,
                    metadata=metadata,
                    detections=detections,
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
                    "No video frames could be read from the input file."
                )

        except BaseException:
            if writer is not None:
                writer.release()

            capture.release()

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

        detection_statistics = DetectionStatistics(
            total_detections=total_detections,
            maximum_vehicles_in_frame=maximum_vehicles_in_frame,
            class_detection_counts=dict(sorted(class_detection_counts.items())),
        )

        summary = ProcessingSummary(
            input_path=input_path,
            output_path=output_path,
            metadata=metadata,
            processed_frames=processed_frames,
            processing_time_seconds=elapsed_seconds,
            average_processing_fps=average_processing_fps,
            model_name=(
                self._detector.model_path if self._detector is not None else None
            ),
            detection_statistics=detection_statistics,
        )

        if summary_path is not None:
            self._write_summary(
                summary=summary,
                summary_path=summary_path.resolve(),
            )

        return summary

    def _detect_vehicles(
        self,
        frame: MatLike,
    ) -> list[Detection]:
        if self._detector is None:
            return []

        return self._detector.detect(frame)

    def _extract_metadata(
        self,
        capture: cv2.VideoCapture,
    ) -> VideoMetadata:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            raise VideoProcessingError(
                "The video contains invalid width or height metadata."
            )

        if fps <= 0:
            raise VideoProcessingError(
                "The video contains invalid or missing FPS metadata."
            )

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=max(frame_count, 0),
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
            writer.release()

            raise VideoProcessingError(
                f"OpenCV could not create the output video: {output_path}"
            )

        return writer

    def _process_frame(
        self,
        frame: MatLike,
        frame_number: int,
        timestamp_seconds: float,
        metadata: VideoMetadata,
        detections: list[Detection],
    ) -> MatLike:
        draw_detections(frame, detections)

        overlay_height = min(
            100,
            max(70, metadata.height // 8),
        )

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
            f"Time: {timestamp_seconds:.2f}s | "
            f"Vehicles: {len(detections)}"
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

        return min(
            (processed_frames / total_frames) * 100,
            100.0,
        )

    @staticmethod
    def _write_summary(
        summary: ProcessingSummary,
        summary_path: Path,
    ) -> None:
        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_path.write_text(
            json.dumps(summary.to_dict(), indent=2),
            encoding="utf-8",
        )
