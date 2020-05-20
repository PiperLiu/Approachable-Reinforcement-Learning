import pygame
import sys
from wrapped_flappy_bird import GameState

game_state = GameState(sound=True)

ACTIONS = [0, 1]

while True:
    action = ACTIONS[0]
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_h:
                action = ACTIONS[1]
    game_state.frame_step(action)
pygame.quit()
