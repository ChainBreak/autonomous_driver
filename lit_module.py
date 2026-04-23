from typing import final
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
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

# Notes:
# - Using an separate parameter just for temperature is fine.

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

        # Collect the tensors from the batch
        frame = batch["frame"]
        action_history = batch["action_history"]
        next_frame = batch["next_frame"]
        next_action_history = batch["next_action_history"]
        expert_action = batch["expert_action"]
        action_category = batch["action_category"]

        # Get the quality of each action given the current state.
        action_values = self.model.quality(frame, action_history)

        # Get the quality of the chosen action.
        chosen_action_value = action_values.gather(
            1, action_category.long().unsqueeze(1)
        ).squeeze(1)

        
        # Estimate the value of the next state
        with torch.no_grad():
            next_state_value: torch.Tensor = self.model_ema.module.soft_state_value(
                frame=next_frame,
                action_history=next_action_history,
            )

        # Compute the implicit reward for each action.
        implicit_reward = chosen_action_value - p.discount_factor * next_state_value

        expert_implicit_reward = implicit_reward[expert_action]
        other_implicit_reward = implicit_reward[~expert_action]

        loss_expert = -expert_implicit_reward.mean()
        loss_other = 0.5 * (other_implicit_reward ** 2).mean()
       

        # calibrate policy temperature
        expert_action_values = action_values[expert_action]
        policy_logits = self.model.action_values_to_policy_logits(expert_action_values.detach())
        loss_policy = F.cross_entropy(policy_logits, action_category[expert_action])
        
        self.log("num expert actions", action_values[expert_action].shape[0], prog_bar=False)

        loss =  loss_expert + loss_other #+ loss_policy
        # self.log_image(frame[0], "frame/current")
        # self.log_image(next_frame[0], "frame/next")
        self.log("train_loss", loss, prog_bar=True)
        self.log("loss/expert_reward", loss_expert, prog_bar=False)
        self.log("loss/other_reward", loss_other, prog_bar=False)
        self.log("loss/policy", loss_policy, prog_bar=False)
        self.log("implicit_reward/expert", expert_implicit_reward.mean(), prog_bar=False)
        self.log("implicit_reward/other", other_implicit_reward.mean(), prog_bar=False)
        self.log("temperature", self.model.log_temperature.exp(), prog_bar=False)
        return loss

    def log_image(self, image: torch.Tensor, name: str) -> None:
        """Log a single image of [C,H,W]"""
        x = image.detach().cpu().float()
        x = torch.clamp(x, 0.0, 1.0)
        self.logger.experiment.add_image(name, x, self.global_step)


# import torch
# import torch.nn.functional as F

# def iq_learn_loss(Q_net, target_Q_net, expert_batch, policy_batch, gamma=0.99):
#     """
#     Q_net:        the Q-network being trained
#     target_Q_net: target network for stable bootstrapping
#     expert_batch: (s, a, s', done) from expert demonstrations
#     policy_batch: (s, a, s', done) from replay buffer / online rollouts
#     """
#     s_e,  a_e,  s_next_e,  done_e  = expert_batch
#     s_p,  a_p,  s_next_p,  done_p  = policy_batch

#     def soft_value(s):
#         """V(s) = logsumexp over actions = E_pi[Q - log pi]"""
#         q = Q_net(s)                        # (B, A)
#         return torch.logsumexp(q, dim=-1)   # (B,)  [tau=1 absorbed into Q scale]

#     def implicit_reward(s, a, s_next, done):
#         """r_hat = Q(s,a) - gamma * V(s')"""
#         q_sa = Q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)  # (B,)
#         with torch.no_grad():
#             v_next = soft_value(s_next)
#         return q_sa - gamma * v_next * (1.0 - done.float())

#     # Implicit reward on each dataset
#     r_expert = implicit_reward(s_e, a_e, s_next_e, done_e)  # push up
#     r_policy = implicit_reward(s_p, a_p, s_next_p, done_p)  # push down

#     # IQ-Learn loss (chi^2 regulariser)
#     loss_expert = -r_expert.mean()               # maximise expert implicit reward
#     loss_policy =  0.5 * (r_policy ** 2).mean()  # chi^2 penalty on policy

#     loss = loss_expert + loss_policy
#     return loss