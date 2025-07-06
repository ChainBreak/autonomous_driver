import lightning as L
from torch.utils.data import DataLoader
import torch
import recorded_dataset
from history_digest import HistoryDigest
from action_categorizer import ActionCategorizer
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
import model
import numpy as np

class LitModule(L.LightningModule):
    """Custom trainer class that extends lightning.Trainer."""
    
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        p = self.hparams

        self.model = model.Model(
            num_action_classes=2**p.action_vector_length,
            action_history_shape=(p.history_digest["num_windows"], p.action_vector_length),
        )

        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, frame: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
        return self.model(frame, action_history)

    def create_history_digest(self) -> HistoryDigest:
        """Create HistoryDigest instance from config parameters."""
        p = self.hparams
        history_digest = HistoryDigest.from_window_growth_rate(
            num_windows=p.history_digest["num_windows"],
            growth_rate=p.history_digest["growth_rate"],
        )
        history_digest.fill(np.zeros(p.action_vector_length))

        return history_digest

    def create_action_categorizer(self) -> ActionCategorizer:
        """Create ActionCategorizer instance from config parameters."""
        p = self.hparams
        return ActionCategorizer(
            action_vector_length=p.action_vector_length
        )

    def create_transform(self) -> transforms.Compose:
        """Create image transforms from config parameters."""
        p = self.hparams
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((p.image_size, p.image_size)),
        ])

    def train_dataloader(self) -> DataLoader:
        print("train_dataloader")
        p = self.hparams

        history_digest = self.create_history_digest()
        action_categorizer = self.create_action_categorizer()
        transform = self.create_transform()

        print(history_digest)

        dataset = recorded_dataset.RecordedDataset(
            data_dir=Path(p.data_dir),
            history_digest=history_digest,
            action_categorizer=action_categorizer,
            transform=transform,
        )

        return DataLoader(
            dataset=dataset, 
            batch_size=p.batch_size,
            shuffle=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        
        frame = batch["frame"]
        action = batch["action"]
        action_category = batch["action_category"]
        action_history = batch["action_history"]

        action_logits = self.model(frame, action_history)
        loss = self.criterion(action_logits, action_category)
        self.log("train_loss", loss)
        return loss
    
