import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROWS = 10
COLS = 17

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 128
MEMORY_SIZE = 100_000
TARGET_UPDATE_EVERY = 1000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 50_000
MAX_GRAD_NORM = 10.0

NUM_FRUIT_TYPES = 6

Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done"]
)


# ============================================================
# Action representation
# ============================================================

@dataclass(frozen=True)
class RectAction:
    r1: int
    c1: int
    r2: int
    c2: int

    def to_array(self) -> np.ndarray:
        return np.array([self.r1, self.c1, self.r2, self.c2], dtype=np.float32)


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


# ============================================================
# State-action encoder
# ============================================================

class StateActionEncoder:
    def __init__(self, rows: int, cols: int, num_fruit_types: int):
        self.rows = rows
        self.cols = cols
        self.num_fruit_types = num_fruit_types
        self.state_dim = rows * cols * num_fruit_types
        self.action_dim = 4
        self.total_dim = self.state_dim + self.action_dim

    def encode_state(self, board: np.ndarray) -> np.ndarray:
        one_hot = np.zeros((self.rows, self.cols, self.num_fruit_types), dtype=np.float32)

        for r in range(self.rows):
            for c in range(self.cols):
                fruit = int(board[r, c])
                if 0 <= fruit < self.num_fruit_types:
                    one_hot[r, c, fruit] = 1.0

        return one_hot.reshape(-1)

    def encode_action(self, action: RectAction) -> np.ndarray:
        return np.array([
            action.r1 / (self.rows - 1),
            action.c1 / (self.cols - 1),
            action.r2 / (self.rows - 1),
            action.c2 / (self.cols - 1),
        ], dtype=np.float32)

    def encode_state_action(self, board: np.ndarray, action: RectAction) -> np.ndarray:
        s = self.encode_state(board)
        a = self.encode_action(action)
        return np.concatenate([s, a], axis=0)


# ============================================================
# Q-network: Q(s, a) -> scalar
# ============================================================

class QNetworkSA(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# Example environment scaffold
# ============================================================

class FruitBoxEnv:
    def __init__(self, rows=ROWS, cols=COLS, num_fruit_types=NUM_FRUIT_TYPES):
        self.rows = rows
        self.cols = cols
        self.num_fruit_types = num_fruit_types
        self.board = None

    def reset(self) -> np.ndarray:
        self.board = np.random.randint(
            0, self.num_fruit_types, size=(self.rows, self.cols), dtype=np.int64
        )
        return self.board.copy()

    def get_state(self) -> np.ndarray:
        return self.board.copy()

    def is_valid_action(self, action: RectAction) -> bool:
        r1, c1, r2, c2 = action.r1, action.c1, action.r2, action.c2

        if not (0 <= r1 <= r2 < self.rows and 0 <= c1 <= c2 < self.cols):
            return False

        area = (r2 - r1 + 1) * (c2 - c1 + 1)
        if area < 2:
            return False

        fruit = self.board[r1, c1]
        rect = self.board[r1:r2 + 1, c1:c2 + 1]
        return np.all(rect == fruit)

    def get_valid_actions(self, max_actions: Optional[int] = None) -> List[RectAction]:
        actions = []
        for r1 in range(self.rows):
            for c1 in range(self.cols):
                for r2 in range(r1, self.rows):
                    for c2 in range(c1, self.cols):
                        action = RectAction(r1, c1, r2, c2)
                        if self.is_valid_action(action):
                            actions.append(action)
                            if max_actions is not None and len(actions) >= max_actions:
                                return actions
        return actions

    def step(self, action: RectAction):
        assert self.is_valid_action(action), f"Invalid action: {action}"

        r1, c1, r2, c2 = action.r1, action.c1, action.r2, action.c2
        area = (r2 - r1 + 1) * (c2 - c1 + 1)
        reward = float(area)

        self.board[r1:r2 + 1, c1:c2 + 1] = -1

        for c in range(self.cols):
            col = self.board[:, c]
            remaining = col[col != -1]
            num_missing = self.rows - len(remaining)
            new_col = np.concatenate([
                -1 * np.ones(num_missing, dtype=np.int64),
                remaining
            ])
            self.board[:, c] = new_col

        mask = (self.board == -1)
        self.board[mask] = np.random.randint(0, self.num_fruit_types, size=mask.sum())

        done = len(self.get_valid_actions(max_actions=1)) == 0
        return self.board.copy(), reward, done


# ============================================================
# Visualization
# ============================================================

class TrainingVisualizer:
    def __init__(self, enabled: bool = True, update_every: int = 10):
        self.enabled = enabled
        self.update_every = update_every

        self.rewards = []
        self.avg_rewards = []
        self.losses = []
        self.epsilons = []

        if self.enabled:
            plt.ion()
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
            self.fig.suptitle("Fruit Box DQN Training Dashboard")

    def moving_average(self, values, window=20):
        if len(values) == 0:
            return []
        out = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            out.append(np.mean(values[start:i+1]))
        return out

    def update(self, episode: int, reward: float, loss: Optional[float], epsilon: float):
        self.rewards.append(reward)
        self.losses.append(np.nan if loss is None else loss)
        self.epsilons.append(epsilon)
        self.avg_rewards = self.moving_average(self.rewards, window=20)

        if not self.enabled:
            return

        if (episode + 1) % self.update_every != 0:
            return

        ax1, ax2, ax3, ax4 = self.axes.flatten()
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()

        ax1.plot(self.rewards)
        ax1.set_title("Episode Reward")
        ax1.set_xlabel("Episode")

        ax2.plot(self.avg_rewards)
        ax2.set_title("Moving Avg Reward (20)")
        ax2.set_xlabel("Episode")

        ax3.plot(self.losses)
        ax3.set_title("Loss")
        ax3.set_xlabel("Episode")

        ax4.plot(self.epsilons)
        ax4.set_title("Epsilon")
        ax4.set_xlabel("Episode")

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def show_board(self, board: np.ndarray, title: str = "Board"):
        if not self.enabled:
            return
        plt.figure(figsize=(8, 4))
        plt.imshow(board, aspect="auto")
        plt.title(title)
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.001)


# ============================================================
# DQN Agent using Q(s, a)
# ============================================================

class DQNStateActionAgent:
    def __init__(
        self,
        rows=ROWS,
        cols=COLS,
        num_fruit_types=NUM_FRUIT_TYPES,
        gamma=GAMMA,
        lr=LR,
    ):
        self.encoder = StateActionEncoder(rows, cols, num_fruit_types)

        self.policy_net = QNetworkSA(self.encoder.total_dim).to(DEVICE)
        self.target_net = QNetworkSA(self.encoder.total_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(MEMORY_SIZE)

        self.gamma = gamma
        self.steps_done = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.steps_done / EPS_DECAY)
        return EPS_START + frac * (EPS_END - EPS_START)

    def batch_q_values_from_encoded_state(
        self,
        encoded_state: np.ndarray,
        actions: List[RectAction],
        network: nn.Module
    ) -> np.ndarray:
        if len(actions) == 0:
            return np.array([], dtype=np.float32)

        action_batch = np.stack(
            [self.encoder.encode_action(a) for a in actions],
            axis=0
        )
        state_batch = np.repeat(encoded_state[None, :], len(actions), axis=0)
        sa_batch = np.concatenate([state_batch, action_batch], axis=1)

        sa_batch_t = torch.tensor(sa_batch, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            q_values = network(sa_batch_t).cpu().numpy()
        return q_values

    def select_action(self, state: np.ndarray, valid_actions: List[RectAction]) -> RectAction:
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available.")

        eps = self.epsilon()
        self.steps_done += 1

        if random.random() < eps:
            return random.choice(valid_actions)

        encoded_state = self.encoder.encode_state(state)
        q_values = self.batch_q_values_from_encoded_state(
            encoded_state, valid_actions, self.policy_net
        )
        best_idx = int(np.argmax(q_values))
        return valid_actions[best_idx]

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self, env_for_action_generation: FruitBoxEnv):
        if len(self.memory) < BATCH_SIZE:
            return None

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*transitions)

        sa_batch = np.stack([
            self.encoder.encode_state_action(s, a)
            for s, a in zip(batch.state, batch.action)
        ], axis=0)

        sa_batch_t = torch.tensor(sa_batch, dtype=torch.float32, device=DEVICE)
        current_q = self.policy_net(sa_batch_t)

        target_q_list = []

        for next_state, reward, done in zip(batch.next_state, batch.reward, batch.done):
            if done:
                target_q_list.append(reward)
                continue

            env_for_action_generation.board = next_state.copy()
            next_valid_actions = env_for_action_generation.get_valid_actions()

            if len(next_valid_actions) == 0:
                target_q_list.append(reward)
                continue

            encoded_next_state = self.encoder.encode_state(next_state)

            next_q_policy = self.batch_q_values_from_encoded_state(
                encoded_next_state, next_valid_actions, self.policy_net
            )
            best_next_idx = int(np.argmax(next_q_policy))
            best_next_action = next_valid_actions[best_next_idx]

            next_sa = np.concatenate([
                encoded_next_state,
                self.encoder.encode_action(best_next_action)
            ], axis=0)
            next_sa_t = torch.tensor(next_sa, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                next_q_target = self.target_net(next_sa_t).item()

            target = reward + self.gamma * next_q_target
            target_q_list.append(target)

        target_q = torch.tensor(target_q_list, dtype=torch.float32, device=DEVICE)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()

        if self.steps_done % TARGET_UPDATE_EVERY == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()


# ============================================================
# Training loop
# ============================================================

def train_dqn_state_action(
    num_episodes: int = 500,
    max_steps_per_episode: int = 500,
    visualize: bool = True,
    plot_update_every: int = 10,
    show_board_every: int = 50,
):
    env = FruitBoxEnv()
    agent = DQNStateActionAgent()
    visualizer = TrainingVisualizer(enabled=visualize, update_every=plot_update_every)

    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        last_loss = None

        for step in range(max_steps_per_episode):
            valid_actions = env.get_valid_actions()

            if len(valid_actions) == 0:
                break

            action = agent.select_action(state, valid_actions)
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            last_loss = agent.train_step(env)

            state = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)

        visualizer.update(
            episode=episode,
            reward=total_reward,
            loss=last_loss,
            epsilon=agent.epsilon()
        )

        if visualize and (episode + 1) % show_board_every == 0:
            visualizer.show_board(state, title=f"Board after Episode {episode+1}")

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(
                f"Episode {episode+1:4d} | "
                f"avg reward (last 10): {avg_reward:.2f} | "
                f"epsilon: {agent.epsilon():.3f} | "
                f"last loss: {last_loss}"
            )

    return agent, episode_rewards


if __name__ == "__main__":
    agent, rewards = train_dqn_state_action(
        num_episodes=500,
        max_steps_per_episode=500,
        visualize=True,
        plot_update_every=10,
        show_board_every=50,
    )