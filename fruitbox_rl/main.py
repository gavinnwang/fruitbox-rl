import collections
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    import pygame
except ModuleNotFoundError:
    pygame = None

# --- Configuration ---
ROWS, COLS = 10, 17
BLOCK_SIZE = 30
FPS = 30
LR = 0.001
GAMMA = 1
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY_STEPS = 20000
MEMORY_SIZE = 20000
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_EVERY = 250
MAX_GRAD_NORM = 1.0
LOG_EVERY = 10
RENDER_EVERY = 10
FIXED_BOARD = True
FIXED_BOARD_SEED = 0


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: tuple[int, int, int, int]
    reward: float
    next_state: np.ndarray
    done: bool
    next_valid_actions: tuple[tuple[int, int, int, int], ...]


# --- 1. The Model Q(s, a) -> scalar ---
class FruitBoxQNetwork(nn.Module):
    def __init__(self, rows, cols, action_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * rows * cols + action_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


class StateActionEncoder:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.action_dim = 4

    def encode_action(self, action):
        r1, c1, r2, c2 = action
        return np.array(
            [
                r1 / (self.rows - 1),
                c1 / (self.cols - 1),
                r2 / (self.rows - 1),
                c2 / (self.cols - 1),
            ],
            dtype=np.float32,
        )

    def encode_state_action(self, state, action):
        return state.astype(np.float32), self.encode_action(action)


# --- 2. The Game Engine ---
class FruitBoxEnv:
    def __init__(self):
        self.all_rects = []
        for r1 in range(ROWS):
            for c1 in range(COLS):
                for r2 in range(r1, ROWS):
                    for c2 in range(c1, COLS):
                        self.all_rects.append((r1, c1, r2, c2))
        self.rect_coords = np.array(self.all_rects, dtype=np.int16)
        self.r1_idx = self.rect_coords[:, 0]
        self.c1_idx = self.rect_coords[:, 1]
        self.r2_idx = self.rect_coords[:, 2] + 1
        self.c2_idx = self.rect_coords[:, 3] + 1
        self.fixed_grid = None
        self.reset()

    def set_fixed_board(self, grid):
        if grid.shape != (ROWS, COLS):
            raise ValueError(f"Expected fixed board shape {(ROWS, COLS)}, got {grid.shape}")
        self.fixed_grid = grid.astype(np.int32).copy()

    def reset(self):
        if self.fixed_grid is not None:
            self.grid = self.fixed_grid.copy()
        else:
            self.grid = np.random.randint(1, 10, size=(ROWS, COLS))
        self.score = 0
        return self.get_state()

    def get_state(self):
        return self.grid.astype(np.float32) / 9.0

    def get_valid_actions(self, state=None):
        if state is None:
            grid = self.grid
        else:
            # Convert normalized state back to integer cell values.
            grid = np.rint(state * 9.0).astype(np.int32)

        integral = np.pad(grid, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
        rect_sums = (
            integral[self.r2_idx, self.c2_idx]
            - integral[self.r1_idx, self.c2_idx]
            - integral[self.r2_idx, self.c1_idx]
            + integral[self.r1_idx, self.c1_idx]
        )
        valid_indices = np.flatnonzero(rect_sums == 10)
        return [self.all_rects[idx] for idx in valid_indices]

    def step(self, action):
        r1, c1, r2, c2 = action
        rect_sum = np.sum(self.grid[r1 : r2 + 1, c1 : c2 + 1])

        if rect_sum == 10:
            cleared_cells = np.count_nonzero(self.grid[r1 : r2 + 1, c1 : c2 + 1])
            self.grid[r1 : r2 + 1, c1 : c2 + 1] = 0
            self.score += cleared_cells
            reward = float(cleared_cells)
        else:
            reward = -1.0

        next_state = self.get_state()
        next_valid_actions = self.get_valid_actions(next_state)
        done = len(next_valid_actions) == 0
        return next_state, reward, done, next_valid_actions


# --- 3. The Trainer ---
class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = StateActionEncoder(ROWS, COLS)
        self.model = FruitBoxQNetwork(ROWS, COLS, self.encoder.action_dim).to(self.device)
        self.target_model = FruitBoxQNetwork(ROWS, COLS, self.encoder.action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = collections.deque(maxlen=MEMORY_SIZE)
        self.steps = 0

    def epsilon(self):
        progress = min(1.0, self.steps / EPS_DECAY_STEPS)
        return EPS_START + progress * (EPS_END - EPS_START)

    def score_actions(self, state, actions, network):
        if not actions:
            return np.array([], dtype=np.float32)

        action_batch = np.stack(
            [self.encoder.encode_action(action) for action in actions],
            axis=0,
        )
        state_batch = np.repeat(state[None, :, :], len(actions), axis=0)
        state_batch_t = torch.tensor(
            state_batch, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        action_batch_t = torch.tensor(
            action_batch, dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            q_values = network(state_batch_t, action_batch_t)

        return q_values.detach().cpu().numpy()

    def select_action(self, state, valid_actions):
        if not valid_actions:
            raise ValueError("No valid actions available")

        if random.random() < self.epsilon():
            return random.choice(valid_actions)

        q_values = self.score_actions(state, valid_actions, self.model)
        return valid_actions[int(np.argmax(q_values))]

    def select_greedy_action(self, state, valid_actions):
        if not valid_actions:
            raise ValueError("No valid actions available")
        q_values = self.score_actions(state, valid_actions, self.model)
        return valid_actions[int(np.argmax(q_values))]

    def store_transition(
        self, state, action, reward, next_state, done, next_valid_actions
    ):
        self.memory.append(
            Transition(
                state,
                action,
                reward,
                next_state,
                done,
                tuple(next_valid_actions),
            )
        )

    def train_step(self):
        if len(self.memory) < MIN_REPLAY_SIZE:
            return None

        batch = random.sample(self.memory, BATCH_SIZE)
        state_batch = np.stack([t.state for t in batch], axis=0)
        action_batch = np.stack(
            [self.encoder.encode_action(t.action) for t in batch], axis=0
        )
        state_batch_t = torch.tensor(
            state_batch, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        action_batch_t = torch.tensor(
            action_batch, dtype=torch.float32, device=self.device
        )
        current_q = self.model(state_batch_t, action_batch_t)

        target_values = []
        for transition in batch:
            if transition.done:
                target_values.append(transition.reward)
                continue

            if not transition.next_valid_actions:
                target_values.append(transition.reward)
                continue

            next_q_policy = self.score_actions(
                transition.next_state, transition.next_valid_actions, self.model
            )
            best_next_action = transition.next_valid_actions[
                int(np.argmax(next_q_policy))
            ]
            next_state_t = torch.tensor(
                transition.next_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(0)
            next_action_t = torch.tensor(
                self.encoder.encode_action(best_next_action),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            with torch.no_grad():
                next_q_target = self.target_model(next_state_t, next_action_t).item()

            target_values.append(transition.reward + GAMMA * next_q_target)

        target_q = torch.tensor(target_values, dtype=torch.float32, device=self.device)

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()

        if self.steps % TARGET_UPDATE_EVERY == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()


def make_fixed_board(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(1, 10, size=(ROWS, COLS), dtype=np.int32)


def evaluate_greedy_policy(agent, base_grid, episodes=10):
    eval_env = FruitBoxEnv()
    eval_env.set_fixed_board(base_grid)
    scores = []

    for _ in range(episodes):
        state = eval_env.reset()
        done = False

        while not done:
            valid_actions = eval_env.get_valid_actions()
            if not valid_actions:
                break

            action = agent.select_greedy_action(state, valid_actions)
            state, _, done, _ = eval_env.step(action)

        scores.append(eval_env.score)

    return float(np.mean(scores))


# --- 4. Visualization ---
def draw_game(screen, env, last_rect=None):
    if pygame is None:
        return

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
        overlay.fill((0, 255, 0, 80))
        screen.blit(overlay, (c1 * BLOCK_SIZE + 50, r1 * BLOCK_SIZE + 50))

    pygame.display.flip()


# --- Main Loop ---
def main():
    screen = None
    clock = None
    if pygame is not None:
        pygame.init()
        screen = pygame.display.set_mode(
            (COLS * BLOCK_SIZE + 100, ROWS * BLOCK_SIZE + 150)
        )
        clock = pygame.time.Clock()

    env = FruitBoxEnv()
    agent = Agent()
    fixed_board = None

    if FIXED_BOARD:
        fixed_board = make_fixed_board(FIXED_BOARD_SEED)
        env.set_fixed_board(fixed_board)
        print(f"Fixed-board mode enabled with seed={FIXED_BOARD_SEED}")
        print(f"Initial valid moves: {len(env.get_valid_actions(env.reset()))}")

    episode = 0
    score_window = collections.deque(maxlen=100)
    loss_window = collections.deque(maxlen=1000)

    while True:
        state = env.reset()
        done = False
        last_rect = None
        render_episode = (
            pygame is not None and episode > 0 and episode % RENDER_EVERY == 0
        )

        while not done:
            if pygame is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

            valid_actions = env.get_valid_actions()
            if not valid_actions:
                done = True
                break

            action = agent.select_action(state, valid_actions)
            last_rect = action

            next_state, reward, done, next_valid_actions = env.step(action)
            agent.store_transition(
                state, action, reward, next_state, done, next_valid_actions
            )
            loss = agent.train_step()
            if loss is not None:
                loss_window.append(loss)

            state = next_state
            agent.steps += 1

            if render_episode and pygame is not None:
                draw_game(screen, env, last_rect)
                clock.tick(FPS)

        episode += 1
        score_window.append(env.score)
        avg_score = np.mean(score_window)
        avg_loss = np.mean(loss_window) if loss_window else float("nan")
        eval_score = (
            evaluate_greedy_policy(agent, fixed_board) if fixed_board is not None else None
        )

        if episode % LOG_EVERY == 0:
            message = (
                f"Episode {episode} | score={env.score} | avg100={avg_score:.2f} "
                f"| epsilon={agent.epsilon():.3f} | replay={len(agent.memory)} | loss={avg_loss:.4f}"
            )
            if eval_score is not None:
                message += f" | greedy_eval={eval_score:.2f}"
            print(message)


if __name__ == "__main__":
    main()
