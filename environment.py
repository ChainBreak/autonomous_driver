import pygame
import numpy as np
import config
import cv2
import dataclasses

class Environment:
    def __init__(self, map_path: str):
        pygame.init()
          
        # Load the map image first to get its dimensions
        self.map_image = pygame.image.load(map_path)
        self.map_width, self.map_height = self.map_image.get_size()
        
        # Create surface matching map dimensions
        self.surface = pygame.Surface((self.map_width, self.map_height))
        
        self.cars: list[Car] = []

    def add_car(self, car: "Car"):
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

    def get_observations(self) -> list[np.ndarray]:
        
        # Convert to numpy array
        surface_np = pygame.surfarray.array3d(surface=self.surface) # (w,h,c)
        surface_np = np.transpose(surface_np, (1, 0, 2)) # (h,w,c)

        # Get views for all cars
        observations = [car.get_observation(surface_np) for car in self.cars]
        
        return observations

class Car:
    def __init__(self, x: float, y: float, angle: float = 0.0, speed: float = 0.0):
        self.x = x
        self.y = y
        self.angle = angle  # in radians
        self.speed = speed
        self.width = config.car_width  # car width in pixels
        self.length = config.car_height  # car length in pixels
        self.view_width = config.view_width
        self.view_height = config.view_height


    def update(self, dt: float):
        # Update position based on speed and angle
        self.x += self.speed * np.cos(self.angle) * dt
        self.y += self.speed * np.sin(self.angle) * dt

    def draw(self, surface: pygame.Surface):
        # Create a rectangle for the car
        car_rect = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        pygame.draw.rect(car_rect, (255, 0, 0), (0, 0, self.length, self.width))
        
        # Rotate the car rectangle
        rotated_car = pygame.transform.rotate(car_rect, -np.degrees(self.angle))
        new_rect = rotated_car.get_rect(center=(self.x, self.y))
        
        # Draw the car on the surface
        surface.blit(rotated_car, new_rect.topleft)

    def get_observation(self, surface: np.ndarray) -> np.ndarray:

        M = cv2.getRotationMatrix2D(
            center=(self.x,self.y),
            angle=np.degrees(self.angle)+180,
            scale=1.0,
        )
        M[0,2] += self.view_width/2 - self.x
        M[1,2] += self.view_height/2 - self.y
        view = cv2.warpAffine(surface,M,(self.view_width,self.view_height))

        observation = Observation(
            view=view,
        )
        return observation
      
@dataclasses.dataclass
class Observation:
    view: np.ndarray

@dataclasses.dataclass
class Action:
    steering: float
    acceleration: float