import pygame
import numpy as np
import config

class Car:
    def __init__(self, x: float, y: float, angle: float = 0.0, velocity: float = 0.0):
        self.x = x
        self.y = y
        self.angle = angle  # in radians
        self.velocity = velocity
        self.width = config.car_width  # car width in pixels
        self.length = config.car_height  # car length in pixels
        self.view_radius = config.view_width / 2  # radius of view in pixels

    def update(self, dt: float):
        # Update position based on velocity and angle
        self.x += self.velocity * np.cos(self.angle) * dt
        self.y += self.velocity * np.sin(self.angle) * dt

    def draw(self, surface: pygame.Surface):
        # Create a rectangle for the car
        car_rect = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        pygame.draw.rect(car_rect, (255, 0, 0), (0, 0, self.length, self.width))
        
        # Rotate the car rectangle
        rotated_car = pygame.transform.rotate(car_rect, -np.degrees(self.angle))
        new_rect = rotated_car.get_rect(center=(self.x, self.y))
        
        # Draw the car on the surface
        surface.blit(rotated_car, new_rect.topleft)

    def get_view(self, surface: pygame.Surface) -> np.ndarray:
        # Convert pygame surface to numpy array
        view_surface = pygame.Surface((self.view_radius * 2, self.view_radius * 2), pygame.SRCALPHA)
        
        # Calculate the area to crop
        x1 = int(self.x - self.view_radius)
        y1 = int(self.y - self.view_radius)
        x2 = int(self.x + self.view_radius)
        y2 = int(self.y + self.view_radius)
        
        # Blit the relevant portion of the main surface
        view_surface.blit(surface, (0, 0), (x1, y1, x2 - x1, y2 - y1))
        
        # Convert to numpy array
        view_array = pygame.surfarray.array3d(view_surface)
        view_array = np.transpose(view_array, (1, 0, 2))
        
        return view_array

class Environment:
    def __init__(self, map_path: str):
        pygame.init()
          
        # Load the map image first to get its dimensions
        self.map_image = pygame.image.load(map_path)
        map_width, map_height = self.map_image.get_size()
        
        # Create surface matching map dimensions
        self.surface = pygame.Surface((map_width, map_height))
        
        self.cars: list[Car] = []

    def add_car(self, car: Car):
        self.cars.append(car)

    def update(self, dt: float):
        # Update all cars
        for car in self.cars:
            car.update(dt)

    def render(self) -> pygame.Surface:
        
        # Draw the map
        self.surface.blit(self.map_image, (0, 0))
        
        # Draw all cars
        for car in self.cars:
            car.draw(self.surface)
        
        return self.surface

    def get_car_views(self) -> list[np.ndarray]:
        
        # Get views for all cars
        views = []
        for car in self.cars:
            view = car.get_view(self.surface)
            views.append(view)
        
        return views

