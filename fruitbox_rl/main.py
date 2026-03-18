import collections
import random

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- Configuration ---
ROWS, COLS = 17, 10
BLOCK_SIZE = 30
FPS = 60
LR = 0.0005
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 2000


# --- 1. The Model (CNN) ---
class FruitBoxCNN(nn.Module):
    def __init__(self, num_actions):
        super(FruitBoxCNN, self).__init__()
        # Input: 1 channel (17x10 grid)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * ROWS * COLS, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# --- 2. The Game Engine ---
class FruitBoxEnv:
    def __init__(self):
        self.reset()
        # Pre-calculate all possible rectangles
        self.all_rects = []
        for r1 in range(ROWS):
            for c1 in range(COLS):
                for r2 in range(r1, ROWS):
                    for c2 in range(c1, COLS):
                        self.all_rects.append((r1, c1, r2, c2))
        self.action_size = len(self.all_rects)

    def reset(self):
        # Fill grid with numbers 1-9
        self.grid = np.random.randint(1, 10, size=(ROWS, COLS))
        self.score = 0
        return self.get_state()

    def get_state(self):
        # Normalize 1-9 to 0.1-0.9 for the NN
        return self.grid.astype(np.float32) / 9.0

    def get_valid_mask(self):
        mask = np.zeros(self.action_size, dtype=np.float32)
        for i, (r1, c1, r2, c2) in enumerate(self.all_rects):
            if np.sum(self.grid[r1 : r2 + 1, c1 : c2 + 1]) == 10:
                mask[i] = 1.0
        return mask

    def step(self, action_idx):
        r1, c1, r2, c2 = self.all_rects[action_idx]
        rect_sum = np.sum(self.grid[r1 : r2 + 1, c1 : c2 + 1])

        if rect_sum == 10:
            # Calculate how many NON-ZERO numbers are being cleared
            cleared_cells = np.count_nonzero(self.grid[r1 : r2 + 1, c1 : c2 + 1])
            self.grid[r1 : r2 + 1, c1 : c2 + 1] = 0
            self.score += cleared_cells
            reward = cleared_cells * 1.0  # Reward proportional to cells cleared
        else:
            reward = -1.0  # Should not happen with masking

        mask = self.get_valid_mask()
        done = not np.any(mask)  # No more valid moves
        return self.get_state(), reward, done


# --- 3. The Trainer ---
class Agent:
    def __init__(self, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FruitBoxCNN(action_size).to(self.device)
        self.target_model = FruitBoxCNN(action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = collections.deque(maxlen=10000)
        self.steps = 0

    def select_action(self, state, mask, epsilon):
        if random.random() < epsilon:
            valid_indices = np.where(mask == 1)[0]
            return random.choice(valid_indices)

        state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        mask_t = torch.FloatTensor(mask).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_t)
            # Masking: set invalid Q-values to -infinity
            masked_q = q_values.masked_fill(mask_t == 0, float("-inf"))
            return torch.argmax(masked_q).item()


# --- 4. Visualization ---
def draw_game(screen, env, last_rect=None):
    screen.fill((30, 30, 30))
    font = pygame.font.SysFont("arial", 20)

    for r in range(ROWS):
        for c in range(COLS):
            val = env.grid[r, c]
            color = (200, 200, 200) if val > 0 else (50, 50, 50)
            rect = pygame.Rect(
                c * BLOCK_SIZE + 50, r * BLOCK_SIZE + 50, BLOCK_SIZE - 2, BLOCK_SIZE - 2
            )
            pygame.draw.rect(screen, (60, 60, 60), rect, 1)
            if val > 0:
                txt = font.render(str(val), True, color)
                screen.blit(txt, (rect.x + 8, rect.y + 2))

    if last_rect:
        r1, c1, r2, c2 = last_rect
        overlay = pygame.Surface(
            ((c2 - c1 + 1) * BLOCK_SIZE, (r2 - r1 + 1) * BLOCK_SIZE), pygame.SRCALPHA
        )
        overlay.fill((0, 255, 0, 80))  # Semi-transparent green
        screen.blit(overlay, (c1 * BLOCK_SIZE + 50, r1 * BLOCK_SIZE + 50))

    pygame.display.flip()


# --- Main Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((COLS * BLOCK_SIZE + 100, ROWS * BLOCK_SIZE + 150))
    env = FruitBoxEnv()
    agent = Agent(env.action_size)

    epsilon = EPS_START
    episode = 0

    while True:
        state = env.reset()
        done = False
        last_rect = None

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            mask = env.get_valid_mask()
            if not np.any(mask):
                break

            action = agent.select_action(state, mask, epsilon)
            last_rect = env.all_rects[action]

            next_state, reward, done = env.step(action)
            state = next_state

            # Decay epsilon
            epsilon = max(EPS_END, EPS_START - agent.steps / EPS_DECAY)
            agent.steps += 1

            draw_game(screen, env, last_rect)
            pygame.time.delay(50)  # Slow down so we can see it

        episode += 1
        print(f"Episode {episode} finished. Score: {env.score}")


if __name__ == "__main__":
    main()
