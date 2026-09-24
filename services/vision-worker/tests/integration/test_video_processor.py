import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_worker.video.processor import (
    VideoProcessingError,
    VideoProcessor,
)


def create_test_video(
    path: Path,
    width: int = 320,
    height: int = 240,
    fps: float = 10.0,
    frame_count: int = 20,
) -> None:
    codec = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(path),
        codec,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create the integration test video.")

    try:
        for frame_number in range(frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            x_position = 20 + (frame_number * 5)

            cv2.rectangle(
                frame,
                (x_position, 120),
                (x_position + 50, 170),
                (0, 180, 255),
                thickness=-1,
            )

            writer.write(frame)
    finally:
        writer.release()


def test_processor_creates_video_and_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    summary_path = tmp_path / "summary.json"

    create_test_video(input_path)

    processor = VideoProcessor()

    summary = processor.process(
        input_path=input_path,
        output_path=output_path,
        summary_path=summary_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert summary_path.exists()

    assert summary.processed_frames == 20
    assert summary.metadata.width == 320
    assert summary.metadata.height == 240
    assert summary.metadata.fps == pytest.approx(10.0, abs=0.1)
    assert summary.metadata.duration_seconds == pytest.approx(2.0, abs=0.1)

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert saved_summary["processed_frames"] == 20
    assert saved_summary["metadata"]["width"] == 320
    assert saved_summary["metadata"]["height"] == 240


def test_processor_rejects_missing_input(tmp_path: Path) -> None:
    processor = VideoProcessor()

    with pytest.raises(
        VideoProcessingError,
        match="Input video file does not exist",
    ):
        processor.process(
            input_path=tmp_path / "missing.mp4",
            output_path=tmp_path / "output.mp4",
        )


def test_processor_rejects_same_input_and_output_path(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.mp4"

    create_test_video(input_path)

    processor = VideoProcessor()

    with pytest.raises(
        VideoProcessingError,
        match="Input and output video paths cannot be the same",
    ):
        processor.process(
            input_path=input_path,
            output_path=input_path,
        )
