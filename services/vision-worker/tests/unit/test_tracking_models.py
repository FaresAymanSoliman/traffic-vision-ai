from vision_worker.detection.models import BoundingBox
from vision_worker.tracking.models import Point, TrackedVehicle


def test_point_converts_to_dictionary() -> None:
    point = Point(x=120, y=340)

    assert point.to_dict() == {
        "x": 120,
        "y": 340,
    }


def test_tracked_vehicle_calculates_center() -> None:
    vehicle = TrackedVehicle(
        track_id=17,
        class_id=2,
        class_name="car",
        confidence=0.91,
        bounding_box=BoundingBox(
            x1=100,
            y1=50,
            x2=300,
            y2=250,
        ),
    )

    assert vehicle.center == Point(x=200, y=150)


def test_tracked_vehicle_calculates_reference_point() -> None:
    vehicle = TrackedVehicle(
        track_id=17,
        class_id=2,
        class_name="car",
        confidence=0.91,
        bounding_box=BoundingBox(
            x1=100,
            y1=50,
            x2=300,
            y2=250,
        ),
    )

    assert vehicle.reference_point == Point(x=200, y=250)


def test_tracked_vehicle_converts_to_dictionary() -> None:
    vehicle = TrackedVehicle(
        track_id=17,
        class_id=2,
        class_name="car",
        confidence=0.91456,
        bounding_box=BoundingBox(
            x1=100,
            y1=50,
            x2=300,
            y2=250,
        ),
    )

    result = vehicle.to_dict()

    assert result["track_id"] == 17
    assert result["class_name"] == "car"
    assert result["confidence"] == 0.9146
    assert result["reference_point"] == {
        "x": 200,
        "y": 250,
    }
