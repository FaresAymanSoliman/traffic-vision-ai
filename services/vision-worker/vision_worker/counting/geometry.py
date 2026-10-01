from vision_worker.tracking.models import Point

EPSILON = 1e-9


def point_side(
        line_start: Point,
        line_end: Point,
        point: Point,
) -> float:
    """
    Calculate which side of a directed line contains the pont.
    """

    return(
        (line_end.x - line_start.x) * (point.y - line_start.y)
        - (line_end.y - line_start.y) * (point.x - line_start.x)
    )


def orientation(
        first: Point,
        second: Point,
        third: Point,
) -> int:
    """
    Return the orientation of three points.

    0: collinear
    1: clockwise
    2: counterclockwise
    """

    value = (
        (second.y - first.y) * (third.x - second.x)
        -(second.x - first.x) * (third.y - second.y)
    ) 

    if abs(value) < EPSILON:
        return 0

    return 1 if value > 0 else 2


def point_on_segment(
        segment_start: Point,
        point: Point,
        segment_end: Point,
) -> bool:
    return(
        min(segment_start.x , segment_end.x)
        <= point.x
        <= max(segment_start.x, segment_end.x)
        and min(segment_start.y, segment_end.y)
        <= point.y
        <= max(segment_start.y, segment_end.y)
    )

def segments_intersect(
        first_start: Point,
        first_end: Point,
        second_start: Point,
        second_end: Point,
) -> bool:
    first_orientation = orientation(
        first_start,
        first_end,
        second_start,
    )

    second_orientation = orientation(
        first_start,
        first_end,
        second_end,
    )

    third_orientation = orientation(
        second_start,
        second_end,
        first_start,
    )

    fourth_orientation = orientation(
        second_start,
        second_end,
        first_end
    )

    if (
        first_orientation != second_orientation
        and third_orientation != fourth_orientation
    ):
        return True

    if (
        first_orientation == 0
        and point_on_segment(
            first_start,
            second_end,
            first_end,
        )
    ):
        return True

    if(
        second_orientation == 0
        and point_on_segment(
            first_start,
            second_end,
            first_end,
        )
    ):
        return True

    if(
        third_orientation == 0
        and point_on_segment(
            second_start,
            first_start,
            second_end,
        )
    ):
        return True

    return(
        fourth_orientation == 0
        and point_on_segment(
            second_start,
            first_end,
            second_end,
        )
    )


def midpoint(first: Point , second: Point) -> Point:
    return Point(
        x = round((first.x + second.x) / 2),
        y = round((first.y + second.y) / 2),
    )


