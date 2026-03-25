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
        frame_history_digest:HistoryDigest,
        action_categorizer:ActionCategorizer,
        transform:Callable = lambda x: x,
        image_size:int = 64,
    ):
        self.data_dir = data_dir
        self.history_digest = history_digest
        self.frame_history_digest = frame_history_digest
        self.action_categorizer = action_categorizer
        self.transform = transform
        self.image_size = image_size

        self.cache_dir = data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        recording_dirs = self.find_all_recording_dirs(self.data_dir)
        self.frame_path_tuples = self.make_list_of_all_training_items(recording_dirs)
        print(f"Found {len(self.frame_path_tuples)} frame path tuples")
        
    def find_all_recording_dirs(self, data_dir:Path):
        recordings = list(data_dir.glob("recording_*"))
        print(f"Found {len(recordings)} recordings: {data_dir}")
        return recordings

    def make_list_of_all_training_items(self, recording_dirs:list[Path]) -> list[tuple]:
        frame_path_tuples = []
        for recording_dir in recording_dirs:
            frame_path_tuples.extend(
                self.preprocess_single_recording(recording_dir),
            )
        return frame_path_tuples

    def cache_hash_tag(self) -> str:
        action_part = "_".join(f"{w.window_size}" for w in self.history_digest.windows)
        frame_part = "_".join(f"{w.window_size}" for w in self.frame_history_digest.windows)
        return f"{action_part}__{frame_part}"

    def preprocess_single_recording(self, recording_dir: Path) -> list[tuple]:
        """Run history digests over ALL frames in order; only frames with training_marker.txt are added to the training list."""
        all_tuples = self.get_all_frames_ordered(recording_dir)
        marked_tuples = [t for t in all_tuples if t[2].exists()]

        print(f"Found {len(marked_tuples)} out of {len(all_tuples)} marked tuples in {recording_dir}")
    
        if self.preprocessing_complete_path(recording_dir).exists():
            print(f"Using cached data for {recording_dir}")
            return marked_tuples

        print(f"Preprocessing {recording_dir}")
        window_sizes_action = [w.window_size for w in self.history_digest.windows]
        digest_action = HistoryDigest(window_sizes_action)

        window_sizes_frame = [w.window_size for w in self.frame_history_digest.windows]
        digest_frame = HistoryDigest(window_sizes_frame)

        zeros_chw = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)

        for i, (frame_path, action_path, _, action_history_path, frame_history_path) in enumerate(all_tuples):
            action = np.load(action_path)

            frame_pil = Image.open(frame_path)
            frame_chw = self.transform(frame_pil).numpy().astype(np.float32)

            if i == 0:
                digest_action.fill(action)
                digest_frame.fill(zeros_chw)

            action_history = digest_action.get_window_averages_numpy()
            action_history_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(action_history_path, action_history)

            frame_history = digest_frame.get_window_averages_numpy()
            np.save(frame_history_path, frame_history)

            digest_action.push(action)
            digest_frame.push(frame_chw)

        self.preprocessing_complete_path(recording_dir).touch()
        return marked_tuples

    def preprocessing_complete_path(self, recording_dir: Path) -> Path:
        """Cache-side path for preprocessing_complete.txt (under this recording in cache)."""
        cached_recording_dir = self.map_path_to_cache_path(recording_dir)
        return cached_recording_dir / "preprocessing_complete.txt"


    def get_all_frames_ordered(self, recording_dir: Path) -> list[tuple[Path, Path, Path, Path, Path]]:
        """All frames in chronological order: (frame, action, marker, action_history_cache, frame_history_cache)."""
        frame_path_list = sorted(list(recording_dir.glob("*_frame.png")))
        action_path_list = [p.with_name(p.name.replace("_frame.png", "_action.npy")) for p in frame_path_list]
        marker_path_list = [p.with_name(p.name.replace("_frame.png", "_training_marker.txt")) for p in frame_path_list]
        history_path_list = [p.with_name(p.name.replace("_frame.png", "_history.npy")) for p in frame_path_list]
        history_path_list = [self.map_path_to_cache_path(p) for p in history_path_list]
        frame_history_path_list = [p.with_name(p.name.replace("_frame.png", "_frame_history.npy")) for p in frame_path_list]
        frame_history_path_list = [self.map_path_to_cache_path(p) for p in frame_history_path_list]
        return list(zip(frame_path_list, action_path_list, marker_path_list, history_path_list, frame_history_path_list))

    def map_path_to_cache_path(self, path:Path):
        relative_path = path.relative_to(self.data_dir)
        hash_str = self.cache_hash_tag()
        cache_path = self.cache_dir / hash_str / relative_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        return cache_path
    
    def __len__(self):
        return len(self.frame_path_tuples)

    def __getitem__(self, index):

        _frame_path, action_path, _, action_history_path, frame_history_path = self.frame_path_tuples[index]
        action = np.load(action_path)
        action_history = np.load(action_history_path)
        frame_history = np.load(frame_history_path)

        action_category = self.action_categorizer.to_category(action)

        return {
            "action": action.astype(np.float32),
            "action_category": action_category.astype(np.float32),
            "action_history": action_history.astype(np.float32),
            "frame_history": frame_history.astype(np.float32),
        }
