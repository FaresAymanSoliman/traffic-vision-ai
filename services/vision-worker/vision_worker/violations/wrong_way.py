import math
from collections import defaultdict, deque

from vision_worker.tracking.models import Point, TrackedVehicle
from vision_worker.violations.models import (
    ViolationEvent,
    ViolationType,
    WrongWayConfig,
)


class WrongWayDetector:
    def __init__(self, config: WrongWayConfig) -> None:
        self._config = config

        allowed_length = math.hypot(
            config.allowed_direction_x,
            config.allowed_direction_y,
        )

        self._normalized_allowed_direction = (
            config.allowed_direction_x / allowed_length,
            config.allowed_direction_y / allowed_length,
        )

        self._position_histories: dict[
            int,
            deque[Point],
        ] = defaultdict(
            lambda: deque(maxlen=config.history_size)
        )

        self._violation_streaks: dict[int, int] = defaultdict(int)
        self._reported_track_ids: set[int] = set()
        self._events: list[ViolationEvent] = []

    @property
    def events(self) -> tuple[ViolationEvent, ...]:
        return tuple(self._events)

    @property
    def reported_track_ids(self) -> set:
        return set(self._reported_track_ids)

    def update(
        self,
        tracked_vehicles: list[TrackedVehicle],
        frame_number: int,
        timestamp_seconds: float,
    ) -> list:
        new_events: list[ViolationEvent] = []

        for vehicle in tracked_vehicles:
            track_id = vehicle.track_id
            history = self._position_histories[track_id]

            history.append(vehicle.reference_point)

            if track_id in self._reported_track_ids:
                continue

            similarity = self._calculate_direction_similarity(
                history
            )

            if similarity is None:
                self._violation_streaks[track_id] = 0
                continue

            if similarity <= self._config.similarity_threshold:
                self._violation_streaks[track_id] += 1
            else:
                self._violation_streaks[track_id] = 0
                continue

            if (
                self._violation_streaks[track_id]
                < self._config.confirmation_observations
            ):
                continue

            event = ViolationEvent(
                event_id=(
                    f"wrong-way-{track_id}-{frame_number}"
                ),
                violation_type=ViolationType.WRONG_WAY,
                track_id=track_id,
                class_name=vehicle.class_name,
                frame_number=frame_number,
                timestamp_seconds=timestamp_seconds,
                reference_point=vehicle.reference_point,
                direction_similarity=similarity,
                confidence=self._calculate_confidence(
                    similarity
                ),
            )

            self._events.append(event)
            self._reported_track_ids.add(track_id)
            new_events.append(event)

        return new_events

    def total_events(self) -> int:
        return len(self._events)

    def counts_by_class(self) -> dict[str, int]:
        counts: dict[str, int] = {}

        for event in self._events:
            counts[event.class_name] = (
                counts.get(event.class_name, 0) + 1
            )

        return dict(sorted(counts.items()))

    def _calculate_direction_similarity(
        self,
        history: deque[Point],
    ) -> float | None:
        if len(history) < 2:
            return None

        first_point = history[0]
        last_point = history[-1]

        movement_x = last_point.x - first_point.x
        movement_y = last_point.y - first_point.y

        displacement = math.hypot(
            movement_x,
            movement_y,
        )

        if (
            displacement
            < self._config.minimum_displacement_pixels
        ):
            return None

        normalized_movement_x = movement_x / displacement
        normalized_movement_y = movement_y / displacement

        allowed_x, allowed_y = (
            self._normalized_allowed_direction
        )

        return (
            normalized_movement_x * allowed_x
            + normalized_movement_y * allowed_y
        )

    @staticmethod
    def _calculate_confidence(
        similarity: float,
    ) -> float:
        return max(
            0.0,
            min(1.0, -similarity),
        )