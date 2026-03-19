from typing import Callable
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
        self.frame_path_tuples = self.make_list_of_all_training_items(recording_dirs)
        
    def find_all_recording_dirs(self, data_dir:Path):
        recordings = list(data_dir.glob("recording_*"))
        print(f"Found {len(recordings)} recordings: {data_dir}")
        return recordings

    def make_list_of_all_training_items(self, recording_dirs:list[Path]):
        frame_path_tuples = []
        for recording_dir in recording_dirs:
            frame_path_tuples.extend(
                self.preprocess_single_recording(recording_dir),
            )
        return frame_path_tuples

    def preprocess_single_recording(self, recording_dir: Path) -> list[tuple[Path, Path, Path]]:
        """Run history digest over ALL frames in order; only frames with training_marker.txt are added to the training list."""
        all_tuples = self.get_all_frames_ordered(recording_dir)
        marked_tuples = [t for t in all_tuples if self.has_training_marker(t[0])]
        if not marked_tuples:
            return []
        if self.check_preprocessing_complete(marked_tuples):
            return marked_tuples

        print(f"Preprocessing {recording_dir}")
        window_sizes = [w.window_size for w in self.history_digest.windows]
        digest = HistoryDigest(window_sizes)
        training_list: list[tuple[Path, Path, Path]] = []

        for i, (frame_path, action_path, history_path) in enumerate(all_tuples):
            action = np.load(action_path)
            if i == 0:
                digest.fill(action)
            if self.has_training_marker(frame_path):
                action_history = digest.get_window_averages_numpy()
                history_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(history_path, action_history)
                training_list.append((frame_path, action_path, history_path))
            digest.push(action)

        return training_list

    def check_preprocessing_complete(self, frame_path_tuples: list[tuple[Path, Path, Path]]) -> bool:
        """Preprocessing is complete if the first and last (marked) history paths exist."""
        if not frame_path_tuples:
            return True
        return frame_path_tuples[0][2].exists() and frame_path_tuples[-1][2].exists()

    def get_all_frames_ordered(self, recording_dir: Path) -> list[tuple[Path, Path, Path]]:
        """All frames in chronological order: (frame_path, action_path, history_path)."""
        frame_path_list = sorted(list(recording_dir.glob("*_frame.png")))
        action_path_list = [p.with_name(p.name.replace("_frame.png", "_action.npy")) for p in frame_path_list]
        history_path_list = [p.with_name(p.name.replace("_frame.png", "_history.npy")) for p in frame_path_list]
        history_path_list = [self.map_path_to_cache_path(p) for p in history_path_list]
        return list(zip(frame_path_list, action_path_list, history_path_list))

    @staticmethod
    def has_training_marker(frame_path: Path) -> bool:
        """True if this frame has a training_marker.txt file (human override)."""
        stem = frame_path.stem.replace("_frame", "")
        marker = frame_path.parent / f"{stem}_training_marker.txt"
        return marker.exists()

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

        frame = self.transform(frame)
        action_category = self.action_categorizer.to_category(action)

        return {
            "frame": frame,
            "action": action.astype(np.float32),
            "action_category": action_category.astype(np.float32),
            "action_history": action_history.astype(np.float32),
        }