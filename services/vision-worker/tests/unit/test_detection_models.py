from vision_worker.detection.models import BoundingBox, Detection


def test_bounding_box_calculates_dimensions() -> None:
    box = BoundingBox(
        x1=100,
        y1=50,
        x2=300,
        y2=200,
    )

    assert box.width == 200
    assert box.height == 150
    assert box.area == 30000
    assert box.center == (200, 125)


def test_bounding_box_prevents_negative_dimensions() -> None:
    box = BoundingBox(
        x1=300,
        y1=200,
        x2=100,
        y2=50,
    )

    assert box.width == 0
    assert box.height == 0
    assert box.area == 0


def test_detection_converts_to_dictionary() -> None:
    detection = Detection(
        class_id=2,
        class_name="car",
        confidence=0.923456,
        bounding_box=BoundingBox(
            x1=10,
            y1=20,
            x2=110,
            y2=120,
        ),
    )

    result = detection.to_dict()

    assert result["class_name"] == "car"
    assert result["confidence"] == 0.9235
    assert result["bounding_box"]["width"] == 100
