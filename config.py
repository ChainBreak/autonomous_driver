from pathlib import Path

project_root = Path(__file__).parent

fps = 30
view_width = 96
view_height = 96
training_width = 64
training_height = 64
view_display_width = 100
view_display_height = 100
num_cars = 10

car_width = 6
car_height = 10 
car_max_speed = 40
car_acceleration = 80
car_deceleration = 10
car_max_steering_ratio = 5 #deg/distance
car_steering_ratio_speed = 20 #deg/distance/second
map_path = project_root / "map-with-roads-in-city-children-road-for-toy-vector-37977821.jpg"

recording_dir = project_root / "recorded_data"