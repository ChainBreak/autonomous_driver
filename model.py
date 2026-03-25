import torch.nn as nn
import torch
import math

class Model(nn.Module):
    def __init__(self,
        num_action_classes: int,
        action_history_shape: tuple[int, int],
        frame_history_shape: tuple[int, int, int, int],
    ):
        super().__init__()
        # frame_history_shape: (N, C, H, W) — batched as [B, N, C, H, W]
        self.num_action_classes = num_action_classes

        self.frame_history_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(math.prod(frame_history_shape), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.action_history_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(math.prod(action_history_shape), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.action_decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_action_classes),
        )

    def forward(self, frame_history: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
        """frame_history: [B, N, C, H, W]; action_history: [B, num_windows, action_dim]"""
        frame_history_x = self.frame_history_encoder(frame_history)
        action_history_x = self.action_history_encoder(action_history)
        x = torch.cat([frame_history_x, action_history_x], dim=1)
        x = self.action_decoder(x)
        return x
