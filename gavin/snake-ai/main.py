import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
import numpy as np
import pygame
from enum import Enum
from collections import namedtuple

# --- Configuration & Constants ---
BLOCK_SIZE = 20
SPEED = 80 
LR = 0.001
LAMBDA = 0.4  # Decay rate for eligibility trace
GAMMA = 0.9   # Discount factor

# RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma, lambd):
        self.lr = lr
        self.gamma = gamma
        self.lambd = lambd
        self.model = model
        # Initialize Eligibility Traces for all model parameters
        self.traces = [torch.zeros_like(p) for p in model.parameters()]

    def reset_traces(self):
        """Call this at the start of every new game."""
        for t in self.traces:
            t.zero_()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        action_idx = np.argmax(action)

        # 1. Calculate TD Error (delta)
        # delta = Reward + gamma * max(Q_next) - Q_current
        self.model.eval()
        with torch.no_grad():
            next_q = self.model(next_state)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(next_q)
        
        self.model.train()
        current_q_values = self.model(state)
        current_q = current_q_values[0][action_idx]
        td_error = target - current_q

        # 2. Get current gradients (d(Q)/d(theta))
        self.model.zero_grad()
        current_q.backward() 
        
        # 3. Update traces and apply updates manually
        # Trace = gamma * lambda * Trace + gradient
        with torch.no_grad():
            for i, p in enumerate(self.model.parameters()):
                # Update the trace for this parameter
                self.traces[i] = self.gamma * self.lambd * self.traces[i] + p.grad
                
                # Apply the TD error update to the parameter weights
                # theta = theta + alpha * td_error * Trace
                p.data += self.lr * td_error * self.traces[i]

# --- Game Engine ---
class SnakeGameAI:
    def __init__(self, w=640, h=480):
        pygame.init()
        self.font = pygame.font.SysFont('arial', 25)
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake RL - Eligibility Traces')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def is_collision(self, pt=None):
        if pt is None: pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else: # [0, 0, 1]
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction == Direction.LEFT: x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN: y += BLOCK_SIZE
        elif self.direction == Direction.UP: y -= BLOCK_SIZE
        self.head = Point(x, y)

    def play_step(self, action):
        self.frame_iteration += 1
        reward = -0.1 # Tiny penalty for every move
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        self._move(action)
        self.snake.insert(0, self.head)
        
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -50
            return reward, game_over, self.score
            
        if self.head == self.food:
            self.score += 1
            reward = 50
            self._place_food()
        else:
            self.snake.pop()
        
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

def get_state(game):
    head = game.snake[0]
    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)
    
    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        (dir_r and game.is_collision(point_r)) or (dir_l and game.is_collision(point_l)) or 
        (dir_u and game.is_collision(point_u)) or (dir_d and game.is_collision(point_d)),
        (dir_u and game.is_collision(point_r)) or (dir_d and game.is_collision(point_l)) or 
        (dir_l and game.is_collision(point_u)) or (dir_r and game.is_collision(point_d)),
        (dir_d and game.is_collision(point_r)) or (dir_u and game.is_collision(point_l)) or 
        (dir_r and game.is_collision(point_u)) or (dir_l and game.is_collision(point_d)),
        dir_l, dir_r, dir_u, dir_d,
        game.food.x < game.head.x, game.food.x > game.head.x, 
        game.food.y < game.head.y, game.food.y > game.head.y 
    ]
    return np.array(state, dtype=int)

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=GAMMA, lambd=LAMBDA)

    def get_action(self, state):
        self.epsilon = max(10, 80 - self.n_games)
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0) # Added unsqueeze
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        state_old = get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = get_state(game)

        # Train with trace
        agent.trainer.train_step(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.trainer.reset_traces() # Crucial: Clear traces at game end
            
            if score > record:
                record = score
                agent.model.save()
            print(f'Game {agent.n_games}, Score {score}, Record: {record}')

if __name__ == '__main__':
    train()
