import pygame
import numpy as np

class LivePlot:
    def __init__(self, mode, colors=None):
        pygame.init()  
        self.display = pygame.display.set_mode(mode)
        self.font    = pygame.font.Font(pygame.font.get_default_font(), 18)
        self.colors  = colors
        if colors == None:
            self.colors = [
                (255, 0, 0),
                (0, 0, 255)
            ]

    def update(self, vecs, limits):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        self.display.fill((255,255,255))

        for i, vec in enumerate(vecs):
            self.draw_vec(vec, self.colors[i], limits)

        pygame.display.update()
        return True

    def draw_vec(self, vec, color, limits):
        if len(vec) == 0:
            return

        vec_limits = (np.min(vec), np.max(vec))

        adj_max = limits[1] - limits[0]
        if adj_max == 0:
            adj_max = 1

        inv_y = 600 * (vec[0] - limits[0]) / adj_max
        if inv_y ==  float('inf'): inv_y = 600
        if inv_y == -float('inf'): inv_y = 0

        prev = (0, int(600 - inv_y))
        for (x, l) in enumerate(vec):
            inv_y = 600 * (l - limits[0]) / adj_max
            if inv_y ==  float('inf'): inv_y = 600
            if inv_y == -float('inf'): inv_y = 0
            curr = (x, int(600 - inv_y))
            pygame.draw.line(self.display, color, prev, curr, 1)
            prev = curr

            if l in limits or l in vec_limits:
                text = self.font.render('{:.05e}'.format(l), True, color)
                pygame.draw.circle(self.display, color, curr, 5)

                text_x = curr[0] + 10
                text_y = curr[1] - text.get_height() / 2

                if text_x + text.get_width() > 800:
                    text_x = curr[0] - 10 - text.get_width()

                if text_y < 0:
                    text_y = 0
                elif text_y + text.get_height() > 600:
                    text_y = 600 - text.get_height()

                self.display.blit(text, (text_x, text_y))

