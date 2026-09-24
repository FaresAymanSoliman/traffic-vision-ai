from vision_worker.detection.models import BoundingBox
from vision_worker.tracking.history import TrackHistory
from vision_worker.tracking.models import TrackedVehicle


def create_vehicle(
    track_id: int,
    class_name: str = "car",
    x: int = 100,
) -> TrackedVehicle:
    return TrackedVehicle(
        track_id=track_id,
        class_id=2,
        class_name=class_name,
        confidence=0.90,
        bounding_box=BoundingBox(
            x1=x,
            y1=100,
            x2=x + 100,
            y2=200,
        ),
    )


def test_history_stores_track_points() -> None:
    history = TrackHistory(maximum_points_per_track=3)

    history.update(
        tracked_vehicles=[create_vehicle(track_id=1, x=100)],
        frame_number=1,
    )

    history.update(
        tracked_vehicles=[create_vehicle(track_id=1, x=120)],
        frame_number=2,
    )

    points = history.get_points(track_id=1)

    assert len(points) == 2
    assert points[0].x == 150
    assert points[1].x == 170


def test_history_limits_points_per_track() -> None:
    history = TrackHistory(maximum_points_per_track=2)

    for frame_number, x_position in enumerate(
        [100, 120, 140],
        start=1,
    ):
        history.update(
            tracked_vehicles=[
                create_vehicle(
                    track_id=1,
                    x=x_position,
                )
            ],
            frame_number=frame_number,
        )

    points = history.get_points(track_id=1)

    assert len(points) == 2
    assert points[0].x == 170
    assert points[1].x == 190


def test_history_counts_unique_track_ids() -> None:
    history = TrackHistory()

    history.update(
        tracked_vehicles=[
            create_vehicle(track_id=1, class_name="car"),
            create_vehicle(track_id=2, class_name="truck"),
        ],
        frame_number=1,
    )

    history.update(
        tracked_vehicles=[
            create_vehicle(track_id=1, class_name="car"),
        ],
        frame_number=2,
    )

    assert history.unique_count() == 2
    assert history.unique_track_ids() == {1, 2}


def test_history_counts_unique_tracks_by_class() -> None:
    history = TrackHistory()

    history.update(
        tracked_vehicles=[
            create_vehicle(track_id=1, class_name="car"),
            create_vehicle(track_id=2, class_name="car"),
            create_vehicle(track_id=3, class_name="truck"),
        ],
        frame_number=1,
    )

    assert history.unique_counts_by_class() == {
        "car": 2,
        "truck": 1,
    }
