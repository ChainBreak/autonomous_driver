from pathlib import Path

from history_digest import HistoryDigest

project_root = Path(__file__).parent

# Must match defaults in config.yaml (history_digest / frame_history_digest) for Recorder window sizing.
_action_digest = HistoryDigest.from_window_growth_rate(num_windows=12, growth_rate=1.5)
_frame_digest = HistoryDigest.from_window_growth_rate(num_windows=12, growth_rate=1.5)

fps = 15
view_width = 96
view_height = 96
training_width = 64
training_height = 64
view_display_width = 100
view_display_height = 100
num_cars = 10

car_width = 6
car_height = 10 
car_max_speed = 30
car_acceleration = 60
car_deceleration = 8
car_max_steering_ratio = 5 #deg/distance
car_steering_ratio_speed = 15 #deg/distance/second
map_path = project_root / "map-with-roads-in-city-children-road-for-toy-vector-37977821.jpg"

recording_dir = project_root / "recorded_data"

# Frames to retain when not recording (warm-up for both digests); max of action and frame digest chain lengths.
recording_digest_window = max(_action_digest.total_length, _frame_digest.total_length)