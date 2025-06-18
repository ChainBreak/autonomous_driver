from environment import Environment, Car, Observation, Action
import numpy as np
import pygame
import config
from recorder import Recorder

class Game:
    screen: pygame.Surface
    clock: pygame.time.Clock
    env: Environment
    keys_pressed: dict[int, bool] 
    running: bool = False

    def __init__(self):
        # self.setup()
        pass

    def setup(self):
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
        actions[0] = human_action
        self.update(actions=actions)
        self.recorder.record(observations[0], actions[0])

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

        self.clock.tick(config.fps)

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
            view_surface = pygame.surfarray.make_surface(observation.view)
            
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

    def get_human_actions(self) -> Action:
        """Get actions from human player (keyboard input)"""

        action = Action(
            left=self.keys_pressed[pygame.K_LEFT],
            right=self.keys_pressed[pygame.K_RIGHT],
            forward=self.keys_pressed[pygame.K_UP],
            backward=self.keys_pressed[pygame.K_DOWN],
        )
            
        return action

    def get_model_actions(self, observations: list[Observation]) -> list[Action]:
        """Get actions from AI model for each car"""
        # For now, return random actions for each car
        # TODO: Replace with actual model predictions
        actions: list[Action] = []
        for _ in observations:
            action = Action(
                left=bool(np.random.randint(2)),
                right=bool(np.random.randint(2)), 
                forward=bool(np.random.randint(2)),
                backward=bool(np.random.randint(2)),
            )
            actions.append(action)
        return actions

        

