from torch.utils.data import Dataset
from pathlib import Path
from history_digest import HistoryDigest
import numpy as np
import json
from PIL import Image
from action_categorizer import ActionCategorizer

class RecordedDataset(Dataset):
    def __init__(self,
        data_dir:Path,
        history_digest:HistoryDigest,
        action_categorizer:ActionCategorizer,
    ):
        self.data_dir = data_dir
        self.history_digest = history_digest
        self.action_categorizer = action_categorizer

        self.cache_dir = data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        recording_dirs = self.find_all_recording_dirs(self.data_dir)
        self.frame_path_tuples = self.make_list_of_all_training_items(recording_dirs)
        
    def find_all_recording_dirs(self, data_dir:Path):
        recordings = list(data_dir.glob("recording_*"))
        print(f"Found {len(recordings)} recordings")
        return recordings

    def make_list_of_all_training_items(self, recording_dirs:list[Path]):
        frame_path_tuples = []
        for recording_dir in recording_dirs:
            frame_path_tuples.extend(
                self.preprocess_single_recording(recording_dir),
            )
        return frame_path_tuples

    def preprocess_single_recording(self, recording_dir:Path):

        frame_path_tuples = self.get_frame_path_tuples(recording_dir)

        if self.check_preprocessing_complete(frame_path_tuples):
            return frame_path_tuples

        print(f"Preprocessing {recording_dir}")
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

        return frame_path_tuples

    def check_preprocessing_complete(self, frame_path_tuples:list[tuple[Path, Path, Path]]):
        """
        Preprocessing is complete if the first and last history paths exist.
        """
        first_history_path = frame_path_tuples[0][2]
        last_history_path = frame_path_tuples[-1][2]
        return first_history_path.exists() and last_history_path.exists()

    def get_frame_path_tuples(self, recording_dir:Path):
        frame_path_list = sorted(list(recording_dir.glob("*_frame.png")))
        action_path_list = [p.with_name(p.name.replace("_frame.png", "_action.npy")) for p in frame_path_list]
        history_path_list = [p.with_name(p.name.replace("_frame.png", "_history.npy")) for p in frame_path_list]
        history_path_list = [self.map_path_to_cache_path(p) for p in history_path_list]
        return list(zip(frame_path_list, action_path_list, history_path_list))

    def map_path_to_cache_path(self, path:Path):
        relative_path = path.relative_to(self.data_dir)
        hash_str = "_".join([f"{window.window_size}" for window in self.history_digest.windows])
        cache_path = self.cache_dir / hash_str / relative_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        return cache_path
    
    def __len__(self):
        return len(self.frame_path_tuples)

    def __getitem__(self, index):

        frame_path, action_path, action_history_path = self.frame_path_tuples[index]
        frame = Image.open(frame_path)
        action = np.load(action_path)
        action_history = np.load(action_history_path)

        action_category = self.action_categorizer.to_category(action)

        return {
            "frame": frame,
            "action": action,
            "action_category": action_category,
            "action_history": action_history,
        }