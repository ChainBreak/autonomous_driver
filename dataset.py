from torch.utils.data import Dataset
from pathlib import Path
from history_digest import HistoryDigest

class RecordedDataset(Dataset):
    def __init__(self, data_dir:Path, history_digest:HistoryDigest):
        self.data_dir = data_dir
        self.history_digest = history_digest

        recording_folders = self.get_recording_folders()
        
    def get_recording_folders(self):
        recordings = list(self.data_dir.glob("recording_*"))
        print(f"Found {len(recordings)} recordings")
        return recordings

    def

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    