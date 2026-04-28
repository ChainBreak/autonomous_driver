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
        action_category = batch["action_category"].long()

        # Estimate the value of the next state
        with torch.no_grad():
            next_policy_logits, next_action_values, next_state_value = self.model_ema.module.policy_and_quality_and_state_value(
                image=next_frame,
                action_history=next_action_history,
            )
        
        policy_logits, action_values, state_value = self.model.policy_and_quality_and_state_value(
            image=frame,
            action_history=action_history,
        )

        next_state_value[expert_action] = 1.0
        next_state_value = next_state_value.clamp(min=0, max=1)
        next_state_value_target =  p.discount_factor * next_state_value

        # Get the quality of the chosen acation
        chosen_action_value = action_values.gather(
            1, action_category.unsqueeze(1)
        ).squeeze(1)

        # Compute the loss for the quality
        loss_quality = F.mse_loss(chosen_action_value, next_state_value_target)

        expert_policy_loss = F.cross_entropy(policy_logits[expert_action], action_category[expert_action])
        other_policy_loss =  F.cross_entropy(policy_logits[~expert_action], action_values[~expert_action].argmax(dim=1))

        value_regularization = 0.01 * (action_values**2).mean()

        loss = loss_quality + 0.001*(expert_policy_loss + other_policy_loss) + value_regularization

        # Compute the accuracy of the policy for the expert and other actions
        policy_correct = policy_logits.argmax(dim=1) == action_category
        expert_policy_accuracy = policy_correct[expert_action].float().mean()
        other_policy_accuracy = policy_correct[~expert_action].float().mean()

        max_action_prob = policy_logits.softmax(dim=1).max(dim=1).values
        expert_max_action_prob = max_action_prob[expert_action].mean()
        other_max_action_prob = max_action_prob[~expert_action].mean()
        
        expert_state_action_value = next_state_value[expert_action].mean()
        other_state_action_value = next_state_value[~expert_action].mean()


      
        self.log("state_action_value/expert", expert_state_action_value, prog_bar=False)
        self.log("state_action_value/other", other_state_action_value, prog_bar=False)
        self.log("policy_accuracy/expert", expert_policy_accuracy, prog_bar=False)
        self.log("policy_accuracy/other", other_policy_accuracy, prog_bar=False)
        self.log("max_action_prob/expert", expert_max_action_prob, prog_bar=False)
        self.log("max_action_prob/other", other_max_action_prob, prog_bar=False)
        self.log("loss/train", loss, prog_bar=True)
        self.log("loss/quality", loss_quality, prog_bar=False)
        self.log("loss/expert_policy", expert_policy_loss, prog_bar=False)
        self.log("loss/other_policy", other_policy_loss, prog_bar=False)
        self.log("loss/value_regularization", value_regularization, prog_bar=False)
        return loss


# # --- Compute TD(0) advantage ---
#     with torch.no_grad():
#         next_values = value_net(next_states).squeeze(-1)  # V(s')
#         targets = rewards + gamma * next_values * (1 - dones)  # r + γV(s')

#     values = value_net(states).squeeze(-1)                # V(s)
#     advantages = (targets - values).detach()              # A = r + γV(s') - V(s)

#     # --- Policy loss ---
#     logits = policy(states)
#     dist = torch.distributions.Categorical(logits=logits)
#     log_probs = dist.log_prob(actions)

#     policy_loss = -(log_probs * advantages).mean()

#     # --- Value loss: train V(s) toward TD target ---
#     value_loss = nn.functional.mse_loss(values, targets)

#     # --- Update ---
#     loss = policy_loss + value_loss