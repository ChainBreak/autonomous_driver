from typing import final
import lightning as L
from torch.utils.data import DataLoader
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import recorded_dataset
from history_digest import HistoryDigest
from action_categorizer import ActionCategorizer
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms
import model
import numpy as np
import itertools
@final
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

        self.model_ema = AveragedModel(
            self.model,
            multi_avg_fn=get_ema_multi_avg_fn(p.ema_decay),
            use_buffers=True,
        )
    
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
        )

    def configure_optimizers(self):
        p = self.hparams
        return torch.optim.Adam(self.model.parameters(), lr=p.learning_rate)

    def training_step(self, batch, batch_idx):
        p = self.hparams
        self.model_ema.update_parameters(self.model)

        
        frame = batch["frame"]
        action_history = batch["action_history"]
        next_frame = batch["next_frame"]
        next_action_history = batch["next_action_history"]
        expert_action = batch["expert_action"]
        action_category = batch["action_category"]

        # Estimate the value of the next state
        next_state_value = self.model_ema.module.estimate_state_value(
            frame=next_frame,
            action_history=next_action_history,
        )

        # Experate state actions get a reward of 1, all other actions get a reward of 0.
        reward = 1*expert_action

        quality_value_target = reward + p.discount_factor * next_state_value

        action_values = self.model.quality(frame, action_history)
        chosen_action_value = action_values.gather(
            1, action_category.long().unsqueeze(1)
        ).squeeze(1)
        loss_quality = F.mse_loss(chosen_action_value, quality_value_target.detach())

        # calibrate policy temperature
        policy_logits = self.model.action_values_to_policy_logits(action_values.detach())
        loss_policy = F.cross_entropy(policy_logits[expert_action], action_category[expert_action])
        
        expert_state_action_value = next_state_value[expert_action].mean()
        other_state_action_value = next_state_value[~expert_action].mean()


        loss = loss_quality + loss_policy
        self.log("train_loss", loss, prog_bar=True)
        self.log("expert_state_action_value", expert_state_action_value, prog_bar=False)
        self.log("other_state_action_value", other_state_action_value, prog_bar=False)
        self.log("temperature", self.model.log_temperature.exp(), prog_bar=False)
        self.log("loss_quality", loss_quality, prog_bar=False)
        self.log("loss_policy", loss_policy, prog_bar=False)
        return loss



