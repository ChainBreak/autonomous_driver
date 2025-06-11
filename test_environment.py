from environment import Environment, Car
import numpy as np
import pygame

import config

def main():
    # Initialize pygame
    pygame.init()
    
    # Create environment with a blank map (you can replace this with your city map)
    env = Environment("map-with-roads-in-city-children-road-for-toy-vector-37977821.jpg")
    
    # Add some cars with different positions and velocities
    cars = [

        Car(x=np.random.randint(0, env.map_width), y=np.random.randint(0, env.map_height), angle=np.pi/2, speed=40),
        Car(x=np.random.randint(0, env.map_width), y=np.random.randint(0, env.map_height), angle=np.pi/3, speed=40),
        Car(x=np.random.randint(0, env.map_width), y=np.random.randint(0, env.map_height), angle=np.pi/4, speed=40),
        Car(x=np.random.randint(0, env.map_width), y=np.random.randint(0, env.map_height), angle=np.pi/5, speed=40),
        Car(x=np.random.randint(0, env.map_width), y=np.random.randint(0, env.map_height), angle=np.pi/6, speed=40),
        Car(x=np.random.randint(0, env.map_width), y=np.random.randint(0, env.map_height), angle=np.pi/7, speed=40),
        Car(x=np.random.randint(0, env.map_width), y=np.random.randint(0, env.map_height), angle=np.pi/8, speed=40),
        Car(x=np.random.randint(0, env.map_width), y=np.random.randint(0, env.map_height), angle=np.pi/9, speed=40),
        Car(x=np.random.randint(0, env.map_width), y=np.random.randint(0, env.map_height), angle=np.pi/10, speed=40),
    ]
    
    for car in cars:
        env.add_car(car)
    
    # Initialize the display
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Autonomous Driver Simulation")
    
    clock = pygame.time.Clock()
    running = True

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Calculate delta time
        dt = clock.tick(config.fps) / 1000.0

        # Update and render
        env.update(dt)
        env.render()
        
        # Get views for all cars
        observations = env.get_observations()

        # Convert numpy arrays to pygame surfaces and display them
        view_width = config.view_width  # Width of each view
        view_height = config.view_height  # Height of each view
        padding = 10  # Padding between views
        
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
            screen.blit(view_surface, (x, y))

        # Update the display
        pygame.display.flip()

    # Clean up pygame
    pygame.quit() 

if __name__ == "__main__":
    main() 