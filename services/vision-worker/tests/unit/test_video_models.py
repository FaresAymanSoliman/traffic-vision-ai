from pathlib import Path

from vision_worker.video.models import ProcessingSummary, VideoMetadata


def test_metadata_calculates_duration() -> None:
    metadata = VideoMetadata(
        width=1920,
        height=1080,
        fps=30.0,
        frame_count=1800,
    )

    assert metadata.duration_seconds == 60.0


def test_metadata_handles_invalid_fps_safely() -> None:
    metadata = VideoMetadata(
        width=1920,
        height=1080,
        fps=0.0,
        frame_count=1800,
    )

    assert metadata.duration_seconds == 0.0


def test_processing_summary_converts_paths_to_strings() -> None:
    metadata = VideoMetadata(
        width=1280,
        height=720,
        fps=25.0,
        frame_count=250,
    )

    summary = ProcessingSummary(
        input_path=Path("input.mp4"),
        output_path=Path("output.mp4"),
        metadata=metadata,
        processed_frames=250,
        processing_time_seconds=5.0,
        average_processing_fps=50.0,
    )

    result = summary.to_dict()

    assert result["input_path"] == "input.mp4"
    assert result["output_path"] == "output.mp4"
    assert result["processed_frames"] == 250
    assert result["metadata"]["duration_seconds"] == 10.0
