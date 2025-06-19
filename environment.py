import pygame
import numpy as np
import config
import cv2
import dataclasses
from pathlib import Path

class Environment:
    def __init__(self, map_path: Path):
        pygame.init()
          
        # Load the map image first to get its dimensions
        self.map_image = pygame.image.load(map_path)
        self.map_width, self.map_height = self.map_image.get_size()
        
        # Create surface matching map dimensions
        self.surface = pygame.Surface((self.map_width, self.map_height))
        
        self.cars: list[Car] = []

    def add_car(self, car: "Car"):
        self.cars.append(car)

    def update(self, actions: list["Action"], dt: float):
        # Update all cars
        for car, action in zip(self.cars, actions):
            car.update(action=action, dt=dt)

    def render(self) -> pygame.Surface:
        
        # Draw the map
        self.surface.blit(self.map_image, (0, 0))
        
        # Draw all cars
        for car in self.cars:
            car.draw(self.surface)
        
        return self.surface

    def get_observations(self) -> list["Observation"]:

        self.render()

        # Convert to numpy array
        surface_np = pygame.surfarray.array3d(surface=self.surface) # (w,h,c)
        surface_np = np.transpose(surface_np, (1, 0, 2)) # (h,w,c)

        # Get views for all cars
        observations = [car.get_observation(surface_np) for car in self.cars]
        
        return observations

class Car:
    def __init__(self,env: "Environment", x: float, y: float, angle_deg: float = 0.0, speed: float = 0.0):
        self.x = x
        self.y = y
        self.angle_deg = angle_deg  
        self.speed = speed
        self.steering_ratio = 0.0
        self.env = env
        self.width = config.car_width  # car width in pixels
        self.length = config.car_height  # car length in pixels
        self.view_width = config.view_width
        self.view_height = config.view_height


    def update(self, action: "Action", dt: float):

        left, right, forward, backward = action

        # Update the speed
        target_speed = config.car_max_speed * forward - config.car_max_speed * backward
        if forward or backward:
            acceleration = config.car_acceleration * dt
        else:
            acceleration = config.car_deceleration * dt
        self.speed += np.clip(target_speed - self.speed, -acceleration, acceleration)

        # Steering ratio connect distance to degrees turned.
        # This is a simple model of the car's steering.
        steering_ratio_target = config.car_max_steering_ratio * right - config.car_max_steering_ratio * left
        steering_ratio_speed = config.car_steering_ratio_speed * dt
        self.steering_ratio += np.clip(steering_ratio_target - self.steering_ratio, -steering_ratio_speed, steering_ratio_speed)

        # Update the angle
        self.angle_deg += self.steering_ratio * self.speed * dt

        # Update position
        self.x += self.speed * np.cos(np.radians(self.angle_deg)) * dt
        self.y += self.speed * np.sin(np.radians(self.angle_deg)) * dt

        # Keep car in bounds
        self.x = np.clip(self.x, 0, self.env.map_width)
        self.y = np.clip(self.y, 0, self.env.map_height)

    def draw(self, surface: pygame.Surface):
        # Create a rectangle for the car
        car_rect = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        pygame.draw.rect(car_rect, (255, 0, 0), (0, 0, self.length, self.width))
        
        # Rotate the car rectangle
        rotated_car = pygame.transform.rotate(car_rect, -self.angle_deg)
        new_rect = rotated_car.get_rect(center=(self.x, self.y))
        
        # Draw the car on the surface
        surface.blit(rotated_car, new_rect.topleft)

    def get_observation(self, surface_image: np.ndarray) -> "Observation":

        M = cv2.getRotationMatrix2D(
            center=(self.x,self.y),
            angle=self.angle_deg+90,
            scale=1.0,
        )
        M[0,2] += self.view_width/2 - self.x
        M[1,2] += self.view_height/2 - self.y
        view = cv2.warpAffine(surface_image,M,(self.view_width,self.view_height))

        observation = Observation(
            view=view,
        )
        return observation
      
@dataclasses.dataclass
class Observation:
    view: np.ndarray

Action = np.ndarray