import math
import statistics
from collections import defaultdict, deque

from vision_worker.calibration.homography import (
    HomographyTransformer,
)
from vision_worker.speed.models import (
    TimedWorldPosition,
    VehicleSpeed,
)
from vision_worker.tracking.models import TrackedVehicle


class SpeedEstimator:
    def __init__(
        self,
        transformer: HomographyTransformer,
        history_seconds: float = 1.0,
        minimum_observation_seconds: float = 0.4,
        minimum_distance_meters: float = 0.5,
        maximum_speed_kmh: float = 180.0,
        smoothing_window: int = 5,
    ) -> None:
        if history_seconds <= 0:
            raise ValueError(
                "Speed history duration must be greater than zero."
            )

        if minimum_observation_seconds <= 0:
            raise ValueError(
                "Minimum observation duration must be greater than zero."
            )

        if minimum_observation_seconds > history_seconds:
            raise ValueError(
                "Minimum observation duration cannot exceed history duration."
            )

        if minimum_distance_meters < 0:
            raise ValueError(
                "Minimum distance cannot be negative."
            )

        if maximum_speed_kmh <= 0:
            raise ValueError(
                "Maximum speed must be greater than zero."
            )

        if smoothing_window <= 0:
            raise ValueError(
                "Smoothing window must be greater than zero."
            )

        self._transformer = transformer
        self._history_seconds = history_seconds
        self._minimum_observation_seconds = minimum_observation_seconds
        self._minimum_distance_meters = minimum_distance_meters
        self._maximum_speed_kmh = maximum_speed_kmh

        self._position_history: dict[
            int,
            deque[TimedWorldPosition],
        ] = defaultdict(deque)

        self._speed_history: dict[
            int,
            deque[float],
        ] = defaultdict(
            lambda: deque(maxlen=smoothing_window)
        )

        self._class_by_track: dict[int, str] = {}
        self._latest_speeds: dict[int, VehicleSpeed] = {}
        self._measured_speeds: list[VehicleSpeed] = []

    @property
    def transformer(self) -> HomographyTransformer:
        return self._transformer

    def update(
        self,
        tracked_vehicles: list[TrackedVehicle],
        timestamp_seconds: float,
    ) -> dict[int, VehicleSpeed]:
        current_track_ids: set[int] = set()

        for vehicle in tracked_vehicles:
            current_track_ids.add(vehicle.track_id)
            self._class_by_track[vehicle.track_id] = vehicle.class_name

            world_position = self._transformer.transform(
                vehicle.reference_point
            )

            history = self._position_history[vehicle.track_id]

            history.append(
                TimedWorldPosition(
                    timestamp_seconds=timestamp_seconds,
                    position=world_position,
                )
            )

            self._trim_history(
                history=history,
                current_timestamp=timestamp_seconds,
            )

            measurement = self._calculate_speed(
                track_id=vehicle.track_id,
                class_name=vehicle.class_name,
                history=history,
            )

            if measurement is not None:
                self._latest_speeds[vehicle.track_id] = measurement
                self._measured_speeds.append(measurement)

        return {
            track_id: speed
            for track_id, speed in self._latest_speeds.items()
            if track_id in current_track_ids
        }

    def latest_speed(
        self,
        track_id: int,
    ) -> VehicleSpeed | None:
        return self._latest_speeds.get(track_id)

    def vehicles_measured(self) -> int:
        return len(self._latest_speeds)

    def average_speed_kmh(self) -> float:
        if not self._latest_speeds:
            return 0.0

        return statistics.fmean(
            measurement.estimated_speed_kmh
            for measurement in self._latest_speeds.values()
        )

    def maximum_speed_kmh(self) -> float:
        if not self._measured_speeds:
            return 0.0

        return max(
            measurement.estimated_speed_kmh
            for measurement in self._measured_speeds
        )

    def averages_by_class(self) -> dict[str, float]:
        grouped_speeds: dict[str, list[float]] = defaultdict(list)

        for track_id, measurement in self._latest_speeds.items():
            class_name = self._class_by_track.get(
                track_id,
                measurement.class_name,
            )

            grouped_speeds[class_name].append(
                measurement.estimated_speed_kmh
            )

        return {
            class_name: round(statistics.fmean(speeds), 2)
            for class_name, speeds in sorted(grouped_speeds.items())
            if speeds
        }

    def _calculate_speed(
        self,
        track_id: int,
        class_name: str,
        history: deque[TimedWorldPosition],
    ) -> VehicleSpeed | None:
        if len(history) < 2:
            return None

        first = history[0]
        last = history[-1]

        elapsed_seconds = (
            last.timestamp_seconds
            - first.timestamp_seconds
        )

        if elapsed_seconds < self._minimum_observation_seconds:
            return None

        distance_meters = math.hypot(
            last.position.x_meters
            - first.position.x_meters,
            last.position.y_meters
            - first.position.y_meters,
        )

        if distance_meters < self._minimum_distance_meters:
            return None

        raw_speed_kmh = (
            distance_meters / elapsed_seconds
        ) * 3.6

        if (
            not math.isfinite(raw_speed_kmh)
            or raw_speed_kmh < 0
            or raw_speed_kmh > self._maximum_speed_kmh
        ):
            return None

        speed_history = self._speed_history[track_id]
        speed_history.append(raw_speed_kmh)

        smoothed_speed_kmh = statistics.median(speed_history)

        return VehicleSpeed(
            track_id=track_id,
            class_name=class_name,
            estimated_speed_kmh=smoothed_speed_kmh,
            observation_seconds=elapsed_seconds,
            distance_meters=distance_meters,
        )

    def _trim_history(
        self,
        history: deque[TimedWorldPosition],
        current_timestamp: float,
    ) -> None:
        oldest_allowed_timestamp = (
            current_timestamp - self._history_seconds
        )

        while (
            len(history) > 2
            and history[0].timestamp_seconds
            < oldest_allowed_timestamp
        ):
            history.popleft()

            