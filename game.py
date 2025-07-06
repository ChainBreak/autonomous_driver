import time
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

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.random_action_start_time = time.time()
        self.random_action = self.generate_random_action()
   

    def setup(self):

        self.load_model(self.checkpoint_path)

        # Initialize pygame
        pygame.init()
        
        # Create environment with a blank map
        self.env = Environment(config.map_path)
        self.recorder = Recorder(config.recording_dir)
        
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
        self.model = LitModule.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.model.to("mps")

        self.action_categorizer = self.model.create_action_categorizer()
        self.transform = self.model.create_transform()

        # Each car has its own history digest
        self.history_digest_for_each_car = [self.model.create_history_digest() for _ in range(config.num_cars)]
        print(self.history_digest_for_each_car[0])

        return self.model

    def run(self):
        self.running = True

        while self.running:
            self.loop()

        # Clean up pygame
        pygame.quit()

    def loop(self):
        observations = self.get_observations()
        self.draw_screen(observations)
        self.handle_events()
        human_action = self.get_human_actions()
        actions = self.get_model_actions(observations)
        modified_human_action = self.inject_random_action_when_enabled(human_action, self.recorder.recording)
        actions[0] = modified_human_action
        self.update(actions=actions)
        self.recorder.record(observations[0], human_action)

    def handle_events(self):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.recorder.toggle_recording()

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

        # Draw red border when recording
        if self.recorder.recording:
            pygame.draw.rect(
                self.screen,
                (255, 0, 0),  # Red color
                (0, 0, self.screen.get_width(), self.screen.get_height()),
                2  # Border thickness
            )

        # Update the display
        pygame.display.flip()

        self.clock.tick(config.fps)
    

    def get_human_actions(self) -> Action:
        """Get actions from human player (keyboard input)"""

        action = np.array([
            self.keys_pressed[pygame.K_LEFT],
            self.keys_pressed[pygame.K_RIGHT],
            self.keys_pressed[pygame.K_UP],
            self.keys_pressed[pygame.K_DOWN],
        ])
            
        return action


    def get_model_actions(self, observations: list[Observation]) -> list[Action]:
        """Get actions from AI model for each car"""

        # Convert view observations to tensors
        views = [self.transform(observation.view) for observation in observations]
        views = torch.stack(views)

        # Convert action histories to tensors
        action_histories = [history_digest.get_window_averages_numpy() for history_digest in self.history_digest_for_each_car]
        action_histories = [torch.from_numpy(action_history) for action_history in action_histories]
        action_histories = torch.stack(action_histories).float()

        # Move tensors to GPU
        views = views.to("mps")
        action_histories = action_histories.to("mps")

        # Get model predictions
        action_logits = self.model(views, action_histories)
        action_probs = torch.softmax(action_logits, dim=1)
        action_categories = torch.multinomial(action_probs, num_samples=1).squeeze(1)

        actions = [self.action_categorizer.to_action(category.item()) for category in action_categories]
        # Update history digests
        for history_digest, action in zip(self.history_digest_for_each_car, actions):
            history_digest.push(action)

        # Return actions
        return actions

    def generate_random_action(self) -> Action:
        return np.array([ bool(np.random.randint(2)) for _ in range(4) ])
        
    def inject_random_action_when_enabled(self, action: Action, enable: bool):

        
        elapsed_time = time.time() - self.random_action_start_time 
        if elapsed_time < config.random_action_on_duration and enable:
            # left, right, forward, backward
            action = action.copy()
            action[0:2] = self.random_action[0:2]
            print("Random action injected", action)

        if elapsed_time > config.random_action_off_duration + config.random_action_on_duration:
            self.random_action_start_time = time.time()
            self.random_action = self.generate_random_action()
    
        return action
        

