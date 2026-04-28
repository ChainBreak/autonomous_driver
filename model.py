import torch.nn as nn
import torchvision.models as models 
import math
import torch

class Model(nn.Module):
    def __init__(self,
        num_action_classes: int,
        action_history_shape: tuple[int, int],
    ):
        super().__init__()

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

        self.quality_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_action_classes),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_action_classes),
        )


    def forward(self, image, action_history):
        return self.policy(image, action_history)

    def encoder(self, image, action_history):
        image_x = self.image_encoder(image)
        action_history_x = self.action_history_encoder(action_history)
        x = torch.cat([image_x, action_history_x], dim=1)
        return x

    def policy_and_quality_and_state_value(self, image, action_history):
        x = self.encoder(image, action_history)
        policy_x = self.policy_head(x)
        quality_x = self.quality_head(x)
        state_value = self.estimate_state_value(quality_x, policy_x)
        return policy_x, quality_x, state_value

    def policy(self, image, action_history):
        x = self.encoder(image, action_history)
        x = self.policy_head(x)
        return x
        
    def quality(self, image, action_history):
        x = self.encoder(image, action_history)
        x = self.quality_head(x)
        return x

    def estimate_state_value(self, action_values: torch.Tensor, policy_logits: torch.Tensor) -> torch.Tensor:
        """
        Estimate the value of the state given the action values.
        """
        return action_values.max(dim=1).values
        # action_probs = torch.softmax(policy_logits, dim=1)
        # return torch.sum(action_values * action_probs, dim=1)
