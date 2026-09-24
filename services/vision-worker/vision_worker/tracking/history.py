from collections import defaultdict, deque

from vision_worker.tracking.models import Point, TrackedVehicle


class TrackHistory:
    def __init__(
        self,
        maximum_points_per_track: int = 30,
        maximum_missing_frames: int = 90,
    ) -> None:
        if maximum_points_per_track <= 1:
            raise ValueError("Maximum points per track must be greater than one.")

        if maximum_missing_frames <= 0:
            raise ValueError("Maximum missing frames must be greater than zero.")

        self._maximum_points_per_track = maximum_points_per_track
        self._maximum_missing_frames = maximum_missing_frames

        self._points: dict[int, deque[Point]] = defaultdict(
            lambda: deque(maxlen=self._maximum_points_per_track)
        )

        self._last_seen_frame: dict[int, int] = {}
        self._class_names: dict[int, str] = {}

    def update(
        self,
        tracked_vehicles: list[TrackedVehicle],
        frame_number: int,
    ) -> None:
        for vehicle in tracked_vehicles:
            self._points[vehicle.track_id].append(vehicle.center)
            self._last_seen_frame[vehicle.track_id] = frame_number
            self._class_names[vehicle.track_id] = vehicle.class_name

        self._remove_stale_tracks(frame_number)

    def get_points(self, track_id: int) -> tuple[Point, ...]:
        points = self._points.get(track_id)

        if points is None:
            return ()

        return tuple(points)

    def unique_track_ids(self) -> set:
        return set(self._class_names)

    def unique_count(self) -> int:
        return len(self._class_names)

    def unique_counts_by_class(self) -> dict[str, int]:
        counts: dict[str, int] = {}

        for class_name in self._class_names.values():
            counts[class_name] = counts.get(class_name, 0) + 1

        return dict(sorted(counts.items()))

    def _remove_stale_tracks(self, current_frame: int) -> None:
        stale_track_ids = [
            track_id
            for track_id, last_seen_frame in self._last_seen_frame.items()
            if current_frame - last_seen_frame > self._maximum_missing_frames
        ]

        for track_id in stale_track_ids:
            self._points.pop(track_id, None)
            self._last_seen_frame.pop(track_id, None)
