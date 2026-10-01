from collections import Counter

from vision_worker.counting.geometry import (
    midpoint,
    point_side,
    segments_intersect,
)

from vision_worker.counting.models import (
    CountEvent,
    CountingLine,
    CountingSnapshot,
    CrossingDirection,
)

from vision_worker.tracking.models import Point , TrackedVehicle

class LineCounter:
    def __init__(
            self,
            counting_line: CountingLine,
            minimum_movement_pixels: float = 2.0,
    ) -> None:
        if minimum_movement_pixels < 0:
            raise ValueError(
                "Minimum movement pixels cannot be negative."
            )

        self._counting_line = counting_line
        self._minimum_movement_pixels = minimum_movement_pixels

        self._previous_points: dict[int, Point] = {}
        self._counted_track_directions: set[
            tuple[int,CrossingDirection]
        ] = set()

        self._counts_by_class: Counter[str] = Counter()
        self._counts_by_direction: Counter[str] = Counter()
        self._events: list[CountEvent] = []


    @property
    def counting_line(self) -> CountingLine:
        return self._counting_line

    @property
    def events(self) -> tuple[CountEvent, ...]:
        return tuple(self._events)


    def update(
            self,
            tracked_vehicles: list[TrackedVehicle],
            frame_number: int,
            timestamp_seconds: float,
    ) -> list:
        new_events: list[CountEvent] = []

        for vehicle in tracked_vehicles:
            current_point = vehicle.reference_point
            previous_point = self._previous_points.get(vehicle.track_id)

            self._previous_points[vehicle.track_id] = current_point

            if previous_point is None:
                continue

            if not self._has_sufficient_movement(
                previous_point,
                current_point
            ):
                continue

            event = self._create_event_if_crossed(
                vehicle = vehicle,
                previous_point = previous_point,
                current_point = current_point,
                frame_number = frame_number,
                timestamp_seconds = timestamp_seconds,
            )

            if event is None: 
                continue

            event_key = (
                event.track_id,
                event.direction,
            )

            if event_key in self._counted_track_directions:
                continue

            self._counted_track_directions.add(event_key)
            self._events.append(event)
            new_events.append(event)

            self._counts_by_class[event.class_name] += 1
            self._counts_by_direction[event.direction.value] += 1 

        return new_events

    def snapshot(self) -> CountingSnapshot:
        return CountingSnapshot(
            total_crossings = len(self._events),
            counts_by_class = dict(
                sorted(self._counts_by_class.items())
            ),
            counts_by_direction = dict(
                sorted(self._counts_by_direction.items())
            ),
        )

    def _create_event_if_crossed(
            self,
            vehicle: TrackedVehicle,
            previous_point: Point,
            current_point: Point,
            frame_number: int,
            timestamp_seconds: float,
    ) -> CountEvent | None:
        line_start = self._counting_line.start
        line_end = self._counting_line.end

        previous_side = point_side(
            line_start,
            line_end,
            previous_point,
        )

        current_side = point_side(
            line_start,
            line_end,
            current_point,
        )

        if previous_side == 0 and current_side == 0:
            return None

        if previous_side * current_side > 0:
            return None

        if not segments_intersect(
            previous_point,
            current_point,
            line_start,
            line_end,
        ):
            return None

        direction = self._resolve_direction(
            previous_side = previous_side,
            current_side = current_side,
        )

        if direction is None:
            return None

        return CountEvent(
            track_id=vehicle.track_id,
            class_name=vehicle.class_name,
            line_id=self._counting_line.line_id,
            direction=direction,
            frame_number=frame_number,
            timestamp_seconds=timestamp_seconds,
            crossing_point=midpoint(
                previous_point,
                current_point,
            ),
        )


    @staticmethod
    def _resolve_direction(
        previous_side: float,
        current_side: float,
    ) -> CrossingDirection | None:
        if previous_side <= 0 < current_side:
            return CrossingDirection.FORWARD

        if previous_side >= 0 > current_side:
            return CrossingDirection.REVERSE

        return None


    def _has_sufficient_movement(
            self,
            previous_point: Point,
            current_point: Point,
    ) -> bool:
        horizontal_distance = current_point.x - previous_point.x
        vertical_distance = current_point.y - previous_point.y

        squared_distance = (
            horizontal_distance**2 
            + vertical_distance**2
        )

        return squared_distance >= self._minimum_movement_pixels**2

    

    
