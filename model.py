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

        self.action_quality_decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_action_classes),
        )

        self.log_temperature = nn.Parameter(torch.tensor(0.0))



    def forward(self, image: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
        return self.policy(image, action_history)

    def policy(self, image, action_history):
        """Get the policy logits for the given state and action history."""

        # The quality return the value of each action given this state
        action_values = self.quality(image, action_history)

        # The policy is derived from the quality values. 
        # Simply scale by the temperature and then normalize to get the policy.
        policy_logits = self.action_values_to_policy_logits(action_values)

        return policy_logits
        
    def quality(self, image, action_history):
        image_x = self.image_encoder(image)
        action_history_x = self.action_history_encoder(action_history)
        x = torch.cat([image_x, action_history_x], dim=1)
        x = self.action_quality_decoder(x)
        return x

    def soft_state_value(self, frame: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
        
        # Get the quality values of the actions.
        action_values = self.quality(frame, action_history)

        action_probs = torch.softmax(self.action_values_to_policy_logits(action_values), dim=-1)


        state_value = (action_probs * action_values).sum(dim=-1)
        print(action_values.mean(),state_value.mean())
      

        return state_value

    def action_values_to_policy_logits(self, action_values: torch.Tensor) -> torch.Tensor:
        return action_values / self.log_temperature.exp()