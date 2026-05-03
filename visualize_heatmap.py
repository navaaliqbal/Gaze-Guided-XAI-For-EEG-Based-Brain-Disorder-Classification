import asyncio
import platform
import pygame
import numpy as np
import json

# Screen dimensions
WIDTH, HEIGHT = 1920, 1080
FPS = 60

gaze_data = []
with open("recordings/synthetic.json", "r") as f:
    data = json.load(f)

    gaze_data = list(map(lambda d: (d['x'], d['y']) ,data["gaze_data"]))

# Example data: array of (x, y) points
x = np.random.rand(100) * 1920 * 0.1
y = np.random.rand(100) * 1080 * 0.1

# Convert points to Pygame-compatible list of tuples
points = gaze_data

def setup():
    global screen
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("Fullscreen Plot")
    screen.fill((255, 255, 255))  # White background

def update_loop():
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            pygame.quit()
            raise SystemExit
    
    screen.fill((255, 255, 255))  # Clear screen
    for point in points:
        pygame.draw.circle(screen, (0, 0, 255), point, 5)  # Draw blue circles, radius 5
    pygame.display.flip()  # Update display

async def main():
    setup()
    while True:
        update_loop()
        await asyncio.sleep(1.0 / FPS)  # Control frame rate

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())