import cv2 
from cv2.typing import MatLike

from vision_worker.counting.line_counter import LineCounter
from vision_worker.counting.models import CountEvent

LINE_COLOR = (0,165,255)
EVENT_COLOR = (0,255,255)

def draw_counting_information(
        frame: MatLike,
        line_counter: LineCounter,
        new_events: list[CountEvent],
) -> MatLike:
    counting_line = line_counter.counting_line

    cv2.line(
        frame,
        (
            counting_line.start.x,
            counting_line.start.y,
        ),

        (
            counting_line.end.x,
            counting_line.end.y,

        ),
        LINE_COLOR,
        thickness=3,
        lineType=cv2.LINE_AA,    
    )

    line_label_position = (
        counting_line.start.x,
        max(counting_line.start.y -12, 24),
    )

    cv2.putText(
        frame,
        f"Count Line: {counting_line.line_id}",
        line_label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        LINE_COLOR,
        2,
        cv2.LINE_AA,
    )

    for event in new_events:
        cv2.circle(
            frame,
            (
                event.crossing_point.x,
                event.crossing_point.y
            ),
            radius=10,
            color=EVENT_COLOR,
            thickness=3,
        )

    snapshot = line_counter.snapshot()

    panel_x = 20
    panel_y = 115
    panel_width = 280
    panel_height = 145

    cv2.rectangle(
        frame,
        (panel_x,panel_y),
        (
            panel_x + panel_width,
            panel_y + panel_height,
        ),
        (15, 23, 42),
        thickness=-1
    )

    lines =[
        f"Crossing: {snapshot.total_crossings}",
        f"Cars: {snapshot.counts_by_class.get('car' , 0)} ",
        f"Trucks: {snapshot.counts_by_class.get('truck',0)}",
        f"Buses: {snapshot.counts_by_class.get('bus' ,0)}"
    ]

    for index , text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (
                panel_x + 14,
                panel_y +28 + index * 25,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (232, 240, 247),
            1,
            cv2.LINE_AA,
        )

    return frame