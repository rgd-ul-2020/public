#!/usr/bin/python3

import pygame

pygame.init()
display = pygame.display.set_mode((800, 600))


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    pygame.display.update()
