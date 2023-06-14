import torch
import torch.nn.functional as F
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

    def get_tensor_state(self):
        tensor = torch.tensor(self.state).unsqueeze(0).to(torch.long)
        one_hot_encoded_tensor = F.one_hot(tensor, num_classes=self.n_colors)
        one_hot_encoded_tensor = one_hot_encoded_tensor.permute(0, 3, 1, 2)
        return one_hot_encoded_tensor.float()

    def is_terminal(self):
        return np.any(self.color_counts == self.width * self.height)

    def flood(self, x, y, old_color, new_color):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return

        if self.state[x, y] != old_color:
            return

        self.state[x, y] = new_color
        self.color_counts[old_color] -= 1
        self.color_counts[new_color] += 1
        self.flood(x + 1, y, old_color, new_color)
        self.flood(x - 1, y, old_color, new_color)
        self.flood(x, y + 1, old_color, new_color)
        self.flood(x, y - 1, old_color, new_color)

    def step(self, action):
        self.value -= 1

        if action != self.state[0, 0]:
            self.flood(0, 0, self.state[0, 0], action)

        self.find_neighbors()
        reward = None
        terminal = self.is_terminal()
        info = {}
        return self.state, reward, terminal, info

    def neighbor_flood(self, x, y, old_color, visited):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return
        if visited[x, y]:
            return
        if self.state[x, y] != old_color:
            self.valid_actions[self.state[x, y]] = 1
            return
        visited[x, y] = 1
        self.neighbor_flood(x + 1, y, old_color, visited)
        self.neighbor_flood(x - 1, y, old_color, visited)
        self.neighbor_flood(x, y + 1, old_color, visited)
        self.neighbor_flood(x, y - 1, old_color, visited)

    def find_neighbors(self):
        visited = np.zeros((self.width, self.height), dtype=np.int8)
        self.valid_actions = np.zeros(self.n_colors, dtype=np.int8)
        self.neighbor_flood(0, 0, self.state[0, 0], visited)

    def reset(self, state=None):
        self.value = 0
        if state is None:
            self.state = np.random.randint(0, self.n_colors, (self.width, self.height))
        else:
            self.state = state
        self.color_counts = np.bincount(self.state.flatten(), minlength=self.n_colors)
        self.find_neighbors()
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

    def copy(self):
        env = FloodEnv(self.width, self.height, self.n_colors)
        env.state = np.copy(self.state)
        env.color_counts = np.copy(self.color_counts)
        env.valid_actions = np.copy(self.valid_actions)
        env.value = self.value
        return env

    def __str__(self):
        return str(self.state)
