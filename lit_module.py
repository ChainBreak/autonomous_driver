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
        

    def train_dataloader(self) -> DataLoader:
        print("train_dataloader")
        p = self.hparams

        history_digest = HistoryDigest.from_window_growth_rate(
            num_windows=p.history_digest["num_windows"],
            growth_rate=p.history_digest["growth_rate"],
        )

        action_categorizer = ActionCategorizer(
            action_vector_length=p.action_vector_length
        )

        transform = transforms.Compose([
            transforms.Resize((p.image_size, p.image_size)),
            transforms.ToTensor(),
        ])

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
    
