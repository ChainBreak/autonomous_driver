from environment import Environment, Car, Observation, Action
import numpy as np
import pygame
import config
from recorder import Recorder
import torch
from pathlib import Path
from lit_module import LitModule


class Game:
    screen: pygame.Surface
    clock: pygame.time.Clock
    env: Environment
    keys_pressed: dict[int, bool]
    running: bool = False
    model: LitModule | None = None
    action_categorizer = None
    transform = None
    action_history_digest_for_each_car = None
    frame_history_digest_for_each_car = None
    autopilot_on: bool = True
    recording_enabled: bool = False
    recording_on: bool = False

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path

    def setup(self):

        self.load_model(self.checkpoint_path)

        # Initialize pygame
        pygame.init()
        
        # Create environment with a blank map
        self.env = Environment(config.map_path)
        self.recorder = Recorder(config.recording_dir, digest_window=config.recording_digest_window)
        
        # Generate all the cars
        for _ in range(config.num_cars):
            car = Car(
                env=self.env,
                x=np.random.randint(0, self.env.map_width), 
                y=np.random.randint(0, self.env.map_height), 
                angle_deg=np.random.uniform(0, 360), 
                speed=np.random.uniform(10, 40), 
            )
            self.env.add_car(car)
        
        # Initialize the display
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Autonomous Driver Simulation")
        
        self.clock = pygame.time.Clock()

    def load_model(self, checkpoint_path: Path):
        if checkpoint_path is None:
            return None
        
        self.model = LitModule.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.model.to("mps")

        self.action_categorizer = self.model.create_action_categorizer()
        self.transform = self.model.create_transform()

        # Each car has its own action and frame history digests
        self.action_history_digest_for_each_car = [self.model.create_action_history_digest() for _ in range(config.num_cars)]
        self.frame_history_digest_for_each_car = [self.model.create_frame_history_digest() for _ in range(config.num_cars)]
        print(self.action_history_digest_for_each_car[0])
        print(self.frame_history_digest_for_each_car[0])


    def run(self):
        self.running = True

        while self.running:
            self.loop()

        # Clean up pygame
        pygame.quit()

    def loop(self):
        observations = self.get_observations()
        self.handle_events()
        human_action = self.get_human_actions()
        if np.any(human_action):
            self.autopilot_on = False
        self.recording_on = self.recording_enabled and not self.autopilot_on
        self.draw_screen(observations)
        actions = self.get_model_actions(observations)
        actions[0] = actions[0] if self.autopilot_on else human_action
        self.update(actions=actions)
        self.recorder.update(observations[0], actions[0], record=self.recording_on)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.recording_enabled = not self.recording_enabled
                if event.key == pygame.K_a:
                    self.autopilot_on = not self.autopilot_on

        self.keys_pressed = pygame.key.get_pressed()

    def update(self, actions: list[Action]):
        dt = 1/config.fps   

        # Update and render
        self.env.update(actions=actions, dt=dt)


    def get_observations(self) -> list[Observation]:
        # Get views for all cars
        observations = self.env.get_observations()

        return observations


    def draw_screen(self, observations: list[Observation]):
        
        self.screen.fill((0, 0, 0))

        # Convert numpy arrays to pygame surfaces and display them
        view_width = config.view_width  # Width of each view
        view_height = config.view_height  # Height of each view
        padding = 10  # Padding between views

        # Draw the view of each car
        for i, observation in enumerate(observations):
            # Convert numpy array to pygame surface
            view = np.transpose(observation.view, (1, 0, 2)) #h,w,c to w,h,c
            view_surface = pygame.surfarray.make_surface(view)
            
            # Calculate position in grid (2x2 layout)
            row = i // 2
            col = i % 2
            x = col * (view_width + padding)
            y = row * (view_height + padding)
            
            # Scale the view to desired size
            view_surface = pygame.transform.scale(view_surface, (view_width, view_height))
            
            # Draw the view
            self.screen.blit(view_surface, (x, y))

        # Draw Recording / Autopilot state labels (grey when off, green when on)
        font = pygame.font.SysFont(None, 24)
        padding = 10
        rec_color = (0, 255, 0) if self.recording_enabled else (128, 128, 128)
        auto_color = (0, 255, 0) if self.autopilot_on else (128, 128, 128)
        rec_text = font.render(
            f"Recording: {'Enabled' if self.recording_enabled else 'Disabled'}",
            True,
            rec_color,
        )
        auto_text = font.render(f"Autopilot: {'ON' if self.autopilot_on else 'OFF'}", True, auto_color)
        right_x = self.screen.get_width() - padding
        rec_x = right_x - rec_text.get_width()
        auto_x = right_x - auto_text.get_width()
        line_gap = 4
        self.screen.blit(rec_text, (rec_x, padding))
        self.screen.blit(
            auto_text,
            (auto_x, padding + rec_text.get_height() + line_gap),
        )

        # Draw red border when actively recording (human + recording mode)
        if self.recording_on:
            pygame.draw.rect(
                self.screen,
                (255, 0, 0),
                (0, 0, self.screen.get_width(), self.screen.get_height()),
                2,
            )

        # Update the display
        pygame.display.flip()

        self.clock.tick(config.fps)
    

    def get_human_actions(self) -> Action:
        """Get actions from human player (keyboard input)"""

        action = np.array([
            self.keys_pressed[pygame.K_LEFT],
            self.keys_pressed[pygame.K_RIGHT],
            self.keys_pressed[pygame.K_UP],#accelerate
            self.keys_pressed[pygame.K_DOWN],#brake
            self.keys_pressed[pygame.K_RSHIFT],#reverse
        ])
            
        return action


    def get_model_actions(self, observations: list[Observation]) -> list[Action]:
        """Get actions from AI model for each car"""
        if self.model is None:
            return [self.generate_random_action() for _ in range(config.num_cars)]

        # Frame history [B, N, C, H, W] and action history from digests (before push)
        frame_histories = [
            torch.from_numpy(fd.get_window_averages_numpy())
            for fd in self.frame_history_digest_for_each_car
        ]
        frame_histories = torch.stack(frame_histories).float()

        action_histories = [
            torch.from_numpy(action_history_digest.get_window_averages_numpy())
            for action_history_digest in self.action_history_digest_for_each_car
        ]
        action_histories = torch.stack(action_histories).float()

        frame_histories = frame_histories.to("mps")
        action_histories = action_histories.to("mps")

        # Get model predictions
        action_logits = self.model(frame_histories, action_histories)
        action_probs = torch.softmax(action_logits, dim=1)
        action_categories = torch.multinomial(action_probs, num_samples=1).squeeze(1)

        actions = [self.action_categorizer.to_action(category.item()) for category in action_categories]
        # Update history digests: actions and current frame (CHW), same order as observations
        for action_history_digest, frame_digest, action, observation in zip(
            self.action_history_digest_for_each_car,
            self.frame_history_digest_for_each_car,
            actions,
            observations,
        ):
            action_history_digest.push(action)
            frame_chw = self.transform(observation.view).numpy().astype(np.float32)
            frame_digest.push(frame_chw)

        return actions

    def generate_random_action(self) -> Action:
        return np.array([bool(np.random.randint(2)) for _ in range(5)])

