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
from collections import defaultdict
import random

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
        for recording_type, state_transitions in self.training_items.items():
            print(f"Found {len(state_transitions)} state transitions for {recording_type}")
        
    def find_all_recording_dirs(self, data_dir:Path):
        recordings = list(data_dir.glob("recording_*"))
        print(f"Found {len(recordings)} recordings: {data_dir}")
        return recordings

    def make_list_of_all_training_items(self, recording_dirs:list[Path]) -> defaultdict[str, list["StateTransition"]]:
        state_transitions_per_recording_type: defaultdict[str, list["StateTransition"]] = defaultdict(list)

        for recording_dir in recording_dirs:
            state_action = self.compute_full_states_for_single_recording(recording_dir)
            self.group_state_transitions_for_recording_types(state_action, state_transitions_per_recording_type)
            
        return state_transitions_per_recording_type

    def compute_full_states_for_single_recording(self, recording_dir: Path) -> list["StateAction"]:
        """Run history digest over ALL frames in order; only frames with _metadata.json are added to the training list."""
        all_items = self.get_all_frames_ordered(recording_dir)

        if self.cache_complete_marker_path(recording_dir).exists():
            print(f"Using cached data for {recording_dir}")
            return all_items

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

        self.cache_complete_marker_path(recording_dir).touch()
        return all_items


    def group_state_transitions_for_recording_types(self, 
        state_action_paths: list["StateAction"],
        state_transitions_per_recording_type:defaultdict[str, list["StateTransition"]],
        ):
        state_actions = state_action_paths[:-1]
        next_state_actions = state_action_paths[1:]
        
        for state_action, next_state_action in zip(state_actions, next_state_actions):
            if state_action.metadata_path.exists():
                with state_action.metadata_path.open(encoding="utf-8") as f:
                    metadata = json.load(f)
                recording_type = metadata["recording_mode"]
                state_transition = StateTransition(state_action, next_state_action)
                state_transitions_per_recording_type[recording_type].append(state_transition)


    def cache_complete_marker_path(self, recording_dir: Path) -> Path:
        """Cache-side path for preprocessing_complete.txt (under this recording in cache)."""
        cached_recording_dir = self.map_path_to_cache_path(recording_dir)
        return cached_recording_dir / "preprocessing_complete.txt"


    def get_all_frames_ordered(self, recording_dir: Path) -> list["StateAction"]:
        """All frames in chronological order."""
        items: list["StateAction"] = []
        for frame_path in sorted(recording_dir.glob("*_frame.png")):
            action_path = frame_path.with_name(frame_path.name.replace("_frame.png", "_action.npy"))
            metadata_path = frame_path.with_name(frame_path.name.replace("_frame.png", "_metadata.json"))
            history_path = frame_path.with_name(frame_path.name.replace("_frame.png", "_history.npy"))
            history_path = self.map_path_to_cache_path(history_path)
            items.append(StateAction(frame_path, action_path, metadata_path, history_path))
        return items

    def map_path_to_cache_path(self, path:Path):
        relative_path = path.relative_to(self.data_dir)
        hash_str = "_".join([f"{window.window_size}" for window in self.history_digest.windows])
        cache_path = self.cache_dir / hash_str / relative_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        return cache_path
    
    def __len__(self):
        return sum(len(transitions) for transitions in self.training_items.values())

    def __getitem__(self, index):

        # Randomly choose a recording mode
        recording_mode = random.choice(list(self.training_items.keys()))

        # Randomly choose a state transition for the chosen recording mode
        state_transitions = self.training_items[recording_mode]
        state_transition = random.choice(state_transitions)

        state_action = state_transition.state_action
        next_state_action = state_transition.next_state_action

        # State 
        frame = Image.open(state_action.frame_path)
        frame = self.transform(frame)
        action_history = np.load(state_action.history_path)

        # Action
        action = np.load(state_action.action_path)

        # Next State
        next_frame = Image.open(next_state_action.frame_path)
        next_frame = self.transform(next_frame)
        next_action_history = np.load(next_state_action.history_path)

        expert_action = recording_mode == "expert"

        action_category = self.action_categorizer.to_category(action)

        return {
            "frame": frame,
            "next_frame": next_frame,
            "action": action.astype(np.float32),
            "action_category": action_category.astype(np.float32),
            "action_history": action_history.astype(np.float32),
            "next_action_history": next_action_history.astype(np.float32),
            "expert_action": torch.tensor(expert_action, dtype=torch.bool),
        }

@dataclass(frozen=True, slots=True)
class StateTransition:
    state_action:"StateAction"
    next_state_action:"StateAction"
    

@dataclass(frozen=True, slots=True)
class StateAction:
    frame_path: Path
    action_path: Path
    metadata_path: Path
    history_path: Path