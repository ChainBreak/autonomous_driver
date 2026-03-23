from environment import Observation, Action
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from collections import deque


class Recorder:
    frame_count: int = 0
    data_dir: Path = Path("")
    recording_dir: Path = Path("")
    digest_window: int

    def __init__(self, output_dir: Path, digest_window: int):
        self.data_dir = output_dir
        self.digest_window = digest_window
        self._frames_to_delete: deque[list[Path]] = deque()

        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_dir = self.data_dir / f"recording_{datetime_str}"
        self.recording_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0
        print(f"Recording directory: {self.recording_dir}")

    def _delete_frames_outside_digest_window(self) -> None:
        while len(self._frames_to_delete) > self.digest_window:
            paths_to_delete = self._frames_to_delete.popleft()
            for p in paths_to_delete:
                if p.exists():
                    p.unlink()

    def update(
        self,
        observation: Observation,
        action: Action,
        *,
        record: bool,
    ) -> None:
        image_path = self.recording_dir / f"{self.frame_count:06d}_frame.png"
        action_path = self.recording_dir / f"{self.frame_count:06d}_action.npy"

        view = cv2.cvtColor(observation.view, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), view)
        np.save(action_path, action)

        if record:
            marker_path = self.recording_dir / f"{self.frame_count:06d}_training_marker.txt"
            marker_path.touch()
            self._frames_to_delete.clear()
        else:
            self._frames_to_delete.append([image_path, action_path])
            self._delete_frames_outside_digest_window()

        self.frame_count += 1
