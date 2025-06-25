from environment import Observation, Action
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

class Recorder:
    recording:bool = False
    frame_count:int = 0
    data_dir:Path = Path("")
    recording_dir:Path = Path("")

    def __init__(self, output_dir:Path):
        self.data_dir = output_dir

    def start_recording(self):
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_dir = self.data_dir / f"recording_{datetime_str}"
        self.recording_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0
        self.recording = True
        print(f"Recording started to {self.recording_dir}")

    def stop_recording(self):
        self.recording = False
        print(f"Recording stopped")

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def record(self, observation:Observation, action:Action):
        if not self.recording:
            return

        image_path = self.recording_dir / f"{self.frame_count:06d}_frame.png"
        action_path = self.recording_dir / f"{self.frame_count:06d}_action.npy"
 
        view = cv2.cvtColor(observation.view, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(image_path), view)

        np.save(action_path, action)

        self.frame_count += 1


  