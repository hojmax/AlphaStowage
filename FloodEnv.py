import pygame
import numpy as np


class FloodEnv:
    def __init__(self, width, height, n_colors):
        self.colors = [
            tuple(np.random.randint(128, 255, size=3)) for _ in range(n_colors)
        ]
        self.scale = 40
        self.width = width
        self.height = height
        self.n_colors = n_colors
        self.screen = None
        self.reset()

    def is_terminal(self):
        return np.any(self.color_counts == self.width * self.height)

    def flood(self, x, y, old_color, new_color):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return
        if self.state[x, y] != old_color:
            self.neighbor_colors[self.state[x, y]] = 1
            return
        self.state[x, y] = new_color
        self.color_counts[old_color] -= 1
        self.color_counts[new_color] += 1
        self.flood(x + 1, y, old_color, new_color)
        self.flood(x - 1, y, old_color, new_color)
        self.flood(x, y + 1, old_color, new_color)
        self.flood(x, y - 1, old_color, new_color)

    def step(self, action):
        if action != self.state[0, 0]:
            self.neighbor_colors[:] = 0
            self.flood(0, 0, self.state[0, 0], action)

        reward = None
        terminal = self.is_terminal()
        info = {}
        return self.state, reward, terminal, info

    def reset(self):
        self.state = np.random.randint(0, self.n_colors, (self.width, self.height))
        self.color_counts = np.bincount(self.state.flatten(), minlength=self.n_colors)
        # Assume all colors are neighbors to start
        self.neighbor_colors = np.ones(self.n_colors, dtype=np.int8)
        return self.state

    def render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Flood")
            self.screen = pygame.display.set_mode(
                (self.width * self.scale, self.height * self.scale)
            )
            self.screen.fill(0)

        self.draw()
        pygame.event.pump()
        pygame.display.flip()

    def draw(self):
        for x in range(self.width):
            for y in range(self.height):
                pygame.draw.rect(
                    self.screen,
                    self.colors[self.state[x, y]],
                    pygame.Rect(x * self.scale, y * self.scale, self.scale, self.scale),
                )


if __name__ == "__main__":
    env = FloodEnv(10, 10, 4)

    moves = [0, 3, 1, 2]
    for move in moves:
        env.step(move)
        env.render()
        pygame.time.wait(1000)
