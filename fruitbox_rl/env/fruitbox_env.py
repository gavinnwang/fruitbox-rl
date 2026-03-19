import numpy as np
import random

from fruitbox_rl.algorithms.rectangle_finder import find_valid_rectangles

class FruitBoxEnv:
    def __init__(self, rows=17, cols=10, seed=None, target=10):
        self.rows = rows
        self.cols = cols
        self.target = target
        self.rng = np.random.default_rng(seed)
        self.board = None
        self.score = 0
        self.moves = 0
        self.reset()

    def reset(self):
        self.board = self.rng.integers(
            1, 10, size=(self.rows, self.cols), dtype=np.int32
        )
        self.score = 0
        self.moves = 0
        return self.board.copy()

    def get_legal_actions(self):
        return find_valid_rectangles(self.board, self.target)

    def step(self, action):
        legal_actions = self.get_legal_actions()
        if action not in legal_actions:
            raise ValueError(f"Illegal action: {action}")

        r1, c1, r2, c2 = action

        removed = np.count_nonzero(self.board[r1 : r2 + 1, c1 : c2 + 1])
        self.board[r1 : r2 + 1, c1 : c2 + 1] = 0

        reward = int(removed)
        self.score += reward
        self.moves += 1

        done = len(self.get_legal_actions()) == 0
        info = {
            "score": self.score,
            "moves": self.moves,
            "reward": reward,
        }

        return self.board.copy(), reward, done, info

    def render(self):
        print(self.board)


if __name__ == "__main__":
    env = FruitBoxEnv(seed=0)

    state = env.reset()
    env.render()

    while True:
        actions = env.get_legal_actions()

        if not actions:
            break

        action = random.choice(actions)
        print("action:", action)

        state, reward, done, info = env.step(action)
        print("reward:", reward, "info:", info)
        env.render()

        if done:
            break

    print("final score:", env.score)
