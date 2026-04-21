from dataclasses import dataclass
from typing import Callable
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from history_digest import HistoryDigest
import numpy as np
from PIL import Image
from action_categorizer import ActionCategorizer

class RecordedDataset(Dataset):
    def __init__(self,
        data_dir:Path,
        history_digest:HistoryDigest,
        action_categorizer:ActionCategorizer,
        transform:Callable = lambda x: x,
    ):
        self.data_dir = data_dir
        self.history_digest = history_digest
        self.action_categorizer = action_categorizer
        self.transform = transform

        self.cache_dir = data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        recording_dirs = self.find_all_recording_dirs(self.data_dir)
        self.training_items = self.make_list_of_all_training_items(recording_dirs)
        print(f"Found {len(self.training_items)} training items")
        
    def find_all_recording_dirs(self, data_dir:Path):
        recordings = list(data_dir.glob("recording_*"))
        print(f"Found {len(recordings)} recordings: {data_dir}")
        return recordings

    def make_list_of_all_training_items(self, recording_dirs:list[Path]) -> list["RecordingFramePaths"]:
        training_items: list["RecordingFramePaths"] = []
        for recording_dir in recording_dirs:
            training_items.extend(
                self.preprocess_single_recording(recording_dir),
            )
        return training_items

    def preprocess_single_recording(self, recording_dir: Path) -> list["RecordingFramePaths"]:
        """Run history digest over ALL frames in order; only frames with _metadata.json are added to the training list."""
        all_items = self.get_all_frames_ordered(recording_dir)
        marked_items = [item for item in all_items if item.metadata_path.exists()]

        print(f"Found {len(marked_items)} out of {len(all_items)} marked items in {recording_dir}")
    
        if self.preprocessing_complete_path(recording_dir).exists():
            print(f"Using cached data for {recording_dir}")
            return marked_items

        print(f"Preprocessing {recording_dir}")
        window_sizes = [w.window_size for w in self.history_digest.windows]
        digest = HistoryDigest(window_sizes)

        for i, item in enumerate(all_items):
            action = np.load(item.action_path)

            # Fill the digest with the first action
            if i == 0:
                digest.fill(action)

            action_history = digest.get_window_averages_numpy()
            item.history_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(item.history_path, action_history)

            digest.push(action)

        self.preprocessing_complete_path(recording_dir).touch()
        return marked_items

    def preprocessing_complete_path(self, recording_dir: Path) -> Path:
        """Cache-side path for preprocessing_complete.txt (under this recording in cache)."""
        cached_recording_dir = self.map_path_to_cache_path(recording_dir)
        return cached_recording_dir / "preprocessing_complete.txt"


    def get_all_frames_ordered(self, recording_dir: Path) -> list["RecordingFramePaths"]:
        """All frames in chronological order."""
        items: list["RecordingFramePaths"] = []
        for frame_path in sorted(recording_dir.glob("*_frame.png")):
            action_path = frame_path.with_name(frame_path.name.replace("_frame.png", "_action.npy"))
            metadata_path = frame_path.with_name(frame_path.name.replace("_frame.png", "_metadata.json"))
            history_path = frame_path.with_name(frame_path.name.replace("_frame.png", "_history.npy"))
            history_path = self.map_path_to_cache_path(history_path)
            items.append(RecordingFramePaths(frame_path, action_path, metadata_path, history_path))
        return items

    def map_path_to_cache_path(self, path:Path):
        relative_path = path.relative_to(self.data_dir)
        hash_str = "_".join([f"{window.window_size}" for window in self.history_digest.windows])
        cache_path = self.cache_dir / hash_str / relative_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        return cache_path
    
    def __len__(self):
        return len(self.training_items)

    def __getitem__(self, index):

        item = self.training_items[index]
        # State 
        frame = Image.open(item.frame_path)
        action_history = np.load(item.history_path)

        # Action
        action = np.load(item.action_path)

        # Next State
        next_frame = Image.open(item.next_frame_path)
        next_action_history = np.load(item.next_history_path)

        with item.metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)
        recording_mode = metadata["recording_mode"]
        expert_action = recording_mode == "expert"

        frame = self.transform(frame)
        action_category = self.action_categorizer.to_category(action)

        return {
            "frame": frame,
            "action": action.astype(np.float32),
            "action_category": action_category.astype(np.float32),
            "action_history": action_history.astype(np.float32),
            "expert_action": torch.tensor(expert_action, dtype=torch.bool),
        }


@dataclass(frozen=True, slots=True)
class RecordingFramePaths:
    frame_path: Path
    action_path: Path
    metadata_path: Path
    history_path: Path