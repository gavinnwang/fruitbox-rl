import numpy as np


class FruitBoxEnv:
    def __init__(self, rows=10, cols=17, seed=None):
        self.rows = rows
        self.cols = cols
        self.rng = np.random.default_rng(seed)
        self.board = self.rng.integers(
            1, 10, size=(self.rows, self.cols), dtype=np.int32
        )
        self.score = 0
        self.moves = 0

    def reset(self):
        self.board = self.rng.integers(
            1, 10, size=(self.rows, self.cols), dtype=np.int32
        )
        self.score = 0
        self.moves = 0
        return self.board.copy()

    def get_legal_actions(self):
        return []

    def step(self, action):
        r1, c1, r2, c2 = action

        removed = 0
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if self.board[r, c] != 0:
                    removed += 1
                    self.board[r, c] = 0

        reward = removed
        self.score += reward
        self.moves += 1

        done = len(self.get_legal_actions()) == 0
        info = {"score": self.score, "moves": self.moves}

        return self.board.copy(), reward, done, info

    def render(self):
        print(self.board)
