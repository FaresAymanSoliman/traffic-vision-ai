import statistics
from collections import Counter

from vision_worker.density.models import (
    DensityConfig,
    DensityLevel,
    DensityMeasurement,
)
from vision_worker.density.occupancy import (
    RoadOccupancyCalculator,
)
from vision_worker.speed.models import VehicleSpeed
from vision_worker.tracking.models import TrackedVehicle


class DensityEstimator:
    def __init__(
        self,
        occupancy_calculator: RoadOccupancyCalculator,
        config: DensityConfig | None = None,
    ) -> None:
        self._occupancy_calculator = occupancy_calculator
        self._config = config or DensityConfig()

        self._measurements: list[DensityMeasurement] = []
        self._level_frame_counts: Counter[str] = Counter()

    @property
    def occupancy_calculator(
        self,
    ) -> RoadOccupancyCalculator:
        return self._occupancy_calculator

    @property
    def config(self) -> DensityConfig:
        return self._config

    def update(
        self,
        tracked_vehicles: list[TrackedVehicle],
        vehicle_speeds: dict[int, VehicleSpeed],
    ) -> DensityMeasurement:
        active_vehicle_count = len(tracked_vehicles)

        occupancy_ratio = (
            self._occupancy_calculator.calculate(
                tracked_vehicles
            )
        )

        average_speed_kmh = self._average_active_speed(
            tracked_vehicles=tracked_vehicles,
            vehicle_speeds=vehicle_speeds,
        )

        level = self._classify(
            active_vehicle_count=active_vehicle_count,
            occupancy_ratio=occupancy_ratio,
            average_speed_kmh=average_speed_kmh,
        )

        measurement = DensityMeasurement(
            level=level,
            active_vehicle_count=active_vehicle_count,
            occupancy_ratio=occupancy_ratio,
            average_estimated_speed_kmh=average_speed_kmh,
        )

        self._measurements.append(measurement)
        self._level_frame_counts[level.value] += 1

        return measurement

    def final_measurement(
        self,
    ) -> DensityMeasurement | None:
        if not self._measurements:
            return None

        return self._measurements[-1]

    def maximum_level(self) -> DensityLevel:
        if any(
            measurement.level == DensityLevel.HIGH
            for measurement in self._measurements
        ):
            return DensityLevel.HIGH

        if any(
            measurement.level == DensityLevel.MEDIUM
            for measurement in self._measurements
        ):
            return DensityLevel.MEDIUM

        return DensityLevel.LOW

    def average_occupancy_ratio(self) -> float:
        if not self._measurements:
            return 0.0

        return statistics.fmean(
            measurement.occupancy_ratio
            for measurement in self._measurements
        )

    def maximum_occupancy_ratio(self) -> float:
        if not self._measurements:
            return 0.0

        return max(
            measurement.occupancy_ratio
            for measurement in self._measurements
        )

    def maximum_active_vehicles(self) -> int:
        if not self._measurements:
            return 0

        return max(
            measurement.active_vehicle_count
            for measurement in self._measurements
        )

    def level_frame_counts(self) -> dict[str, int]:
        return {
            level.value: self._level_frame_counts.get(
                level.value,
                0,
            )
            for level in DensityLevel
        }

    def _classify(
        self,
        active_vehicle_count: int,
        occupancy_ratio: float,
        average_speed_kmh: float | None,
    ) -> DensityLevel:
        high_by_count = (
            active_vehicle_count
            >= self._config.high_vehicle_count
        )

        high_by_occupancy = (
            occupancy_ratio
            >= self._config.high_occupancy_ratio
        )

        congested_by_speed = (
            active_vehicle_count
            >= self._config.minimum_vehicles_for_speed_rule
            and average_speed_kmh is not None
            and average_speed_kmh
            <= self._config.low_speed_threshold_kmh
            and occupancy_ratio
            >= self._config.medium_occupancy_ratio
        )

        if (
            high_by_count
            or high_by_occupancy
            or congested_by_speed
        ):
            return DensityLevel.HIGH

        medium_by_count = (
            active_vehicle_count
            >= self._config.medium_vehicle_count
        )

        medium_by_occupancy = (
            occupancy_ratio
            >= self._config.medium_occupancy_ratio
        )

        if medium_by_count or medium_by_occupancy:
            return DensityLevel.MEDIUM

        return DensityLevel.LOW

    @staticmethod
    def _average_active_speed(
        tracked_vehicles: list[TrackedVehicle],
        vehicle_speeds: dict[int, VehicleSpeed],
    ) -> float | None:
        active_track_ids = {
            vehicle.track_id
            for vehicle in tracked_vehicles
        }

        active_speeds = [
            measurement.estimated_speed_kmh
            for track_id, measurement in vehicle_speeds.items()
            if track_id in active_track_ids
        ]

        if not active_speeds:
            return None

        return statistics.fmean(active_speeds)