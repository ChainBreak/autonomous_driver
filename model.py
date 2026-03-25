import torch.nn as nn
import torchvision.models as models 
import math

import torch

class Model(nn.Module):
    def __init__(self,
        num_action_classes: int,
        action_history_shape: tuple[int, int],
        frame_history_shape: tuple[int, int, int, int],
    ):
        super().__init__()
        # frame_history_shape: (N, C, H, W) — batched as [B, N, C, H, W]

        # Load pretrained ResNet18 and modify the final layer
        self.image_encoder = models.resnet18()#weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Replace the final fully connected layer to output 512 features instead of 1000 classes
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, 512)

        self.action_history_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(math.prod(action_history_shape), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        resnet_dim = self.image_encoder.fc.out_features
        frame_concat_dim = frame_history_shape[0] * resnet_dim
        decoder_in = frame_concat_dim + 512

        self.action_decoder = nn.Sequential(
            nn.Linear(decoder_in, 512),
            nn.ReLU(),
            nn.Linear(512, num_action_classes),
        )

    def forward(self, frame_history: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
        """frame_history: [B, N, C, H, W]; action_history: [B, num_windows, action_dim]"""

        B, N, C, H, W = frame_history.shape
        frame_history = frame_history.reshape(B * N, C, H, W)
        frame_history_x = self.image_encoder(frame_history)
        # Concatenate all N window encodings: [B, N * resnet_dim]
        frame_history_x = frame_history_x.reshape(B, -1)
        action_history_x = self.action_history_encoder(action_history)
        x = torch.cat([frame_history_x, action_history_x], dim=1)
        x = self.action_decoder(x)
        return x
