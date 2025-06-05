def main():
    print("Hello, World!")


    import pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    sprite = pygame.image.load("red_car.png").convert_alpha()  # Load sprite with transparency
    # Resize sprite to 32x32
    sprite = pygame.transform.scale(sprite, (32, 32))
    
    # Set position and rotation
    x, y = 200, 150  # Specific x,y coordinates
    angle = 45
    
    # Rotate sprite
    rotated_sprite = pygame.transform.rotate(sprite, angle)
    # Get the rect and set its center to our desired position
    sprite_rect = rotated_sprite.get_rect(center=(x, y))
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill((255, 255, 255))  # White background
        screen.blit(rotated_sprite, sprite_rect)  # Draw rotated sprite at x,y
        pygame.display.flip()
        
    pygame.quit()

if __name__ == "__main__":
    main()