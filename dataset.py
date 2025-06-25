from torch.utils.data import Dataset
from pathlib import Path
from history_digest import HistoryDigest
import numpy as np
import json

class RecordedDataset(Dataset):
    def __init__(self, data_dir:Path, history_digest:HistoryDigest):
        self.data_dir = data_dir
        self.history_digest = history_digest

        self.cache_dir = data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        recording_dirs = self.find_all_recording_dirs(self.data_dir)
        self.preprocess_all_recordings(recording_dirs)
        
    def find_all_recording_dirs(self, data_dir:Path):
        recordings = list(data_dir.glob("recording_*"))
        print(f"Found {len(recordings)} recordings")
        return recordings

    def preprocess_all_recordings(self, recording_dirs:list[Path]):
        for recording_dir in recording_dirs:
            self.preprocess_single_recording(recording_dir)

    def preprocess_single_recording(self, recording_dir:Path):
    
        manifest_path = recording_dir / "PREPROCESSED"

        if manifest_path.exists():
            return

        print(f"Preprocessing {recording_dir}")

        frame_path_tuples = self.get_frame_path_tuples(recording_dir)


        for i, (frame_path, action_path, history_path) in enumerate(frame_path_tuples):
            action = np.load(action_path)

            if i == 0:
                # Fill the history digest with the first action
                self.history_digest.fill(action)

            # Get the history digest and save it
            action_history = self.history_digest.get_window_averages_numpy()
            np.save(history_path, action_history)
            
            # Push the action to the history digest
            self.history_digest.push(action)

        with open(manifest_path, "w") as f:
            f.write(f"Frame count: {len(frame_path_tuples)}")
  

    def get_frame_path_tuples(self, recording_dir:Path):
        frame_path_list = sorted(list(recording_dir.glob("*_frame.png")))
        action_path_list = [p.with_name(p.name.replace("_frame.png", "_action.npy")) for p in frame_path_list]
        history_path_list = [p.with_name(p.name.replace("_frame.png", "_history.npy")) for p in frame_path_list]
        return list(zip(frame_path_list, action_path_list, history_path_list))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    