import pygame
import numpy as np

def create_blank_map(width=800, height=600):
    # Create a blank surface
    surface = pygame.Surface((width, height))
    
    # Fill with white background
    surface.fill((255, 255, 255))
    
    # Draw some simple roads
    pygame.draw.rect(surface, (100, 100, 100), (0, height//2 - 20, width, 40))  # horizontal road
    pygame.draw.rect(surface, (100, 100, 100), (width//2 - 20, 0, 40, height))  # vertical road
    
    # Save the map
    pygame.image.save(surface, "blank_map.png")

if __name__ == "__main__":
    pygame.init()
    create_blank_map()
    pygame.quit() 