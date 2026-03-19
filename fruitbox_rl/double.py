import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pygame
import collections
import csv
import os

# --- Configuration ---
ROWS, COLS = 10, 17
BLOCK_SIZE = 30
LR = 0.0003
GAMMA = 0.99
MEMORY_SIZE = 50_000
BATCH_SIZE = 64
TARGET_UPDATE = 1000 
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 20000 
LOG_EVERY = 10
RENDER_EVERY = 100
CSV_FILE = "double_dqn_log.csv"

# --- 1. The Model (CNN) ---
class FruitBoxCNN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * ROWS * COLS, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- 2. The Optimized Environment ---
class FruitBoxEnv:
    def __init__(self):
        self.all_rects = []
        for r1 in range(ROWS):
            for c1 in range(COLS):
                for r2 in range(r1, ROWS):
                    for c2 in range(c1, COLS):
                        self.all_rects.append((r1, c1, r2, c2))
        
        self.action_size = len(self.all_rects)
        rects = np.array(self.all_rects)
        self.r1, self.c1, self.r2, self.c2 = rects[:,0], rects[:,1], rects[:,2]+1, rects[:,3]+1
        self.reset()

    def reset(self, seed=None):
        if seed is not None: np.random.seed(seed)
        self.grid = np.random.randint(1, 10, size=(ROWS, COLS))
        self.score = 0
        return self.get_state()

    def get_state(self):
        return self.grid.astype(np.float32) / 9.0

    def get_valid_mask(self):
        integral = np.pad(self.grid, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        sums = (integral[self.r2, self.c2] - integral[self.r1, self.c2] - 
                integral[self.r2, self.c1] + integral[self.r1, self.c1])
        return (sums == 10)

    def step(self, action_idx):
        r1, c1, r2, c2 = self.all_rects[action_idx]
        cleared = np.count_nonzero(self.grid[r1:r2+1, c1:c2+1])
        self.grid[r1:r2+1, c1:c2+1] = 0
        self.score += cleared
        
        mask = self.get_valid_mask()
        done = not np.any(mask)
        return self.get_state(), float(cleared), done, mask

# --- 3. The Double DQN Agent ---
class Agent:
    def __init__(self, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = FruitBoxCNN(action_size).to(self.device)
        self.target_net = FruitBoxCNN(action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = collections.deque(maxlen=MEMORY_SIZE)
        self.steps = 0

    def select_action(self, state, mask, epsilon, greedy=False):
        if not greedy and random.random() < epsilon:
            valid_indices = np.where(mask == True)[0]
            return random.choice(valid_indices)
        
        state_t = torch.as_tensor(state, device=self.device).view(1, 1, ROWS, COLS)
        with torch.no_grad():
            q_vals = self.policy_net(state_t).squeeze()
            q_vals[torch.as_tensor(~mask, device=self.device)] = -1e9
            return q_vals.argmax().item()

    def train_step(self):
        if len(self.memory) < BATCH_SIZE: return 0
        
        batch = random.sample(self.memory, BATCH_SIZE)
        s, a, r, ns, nm, d = zip(*batch)

        s_t = torch.as_tensor(np.array(s), device=self.device).unsqueeze(1)
        a_t = torch.as_tensor(a, device=self.device).unsqueeze(1)
        r_t = torch.as_tensor(r, device=self.device, dtype=torch.float)
        ns_t = torch.as_tensor(np.array(ns), device=self.device).unsqueeze(1)
        nm_t = torch.as_tensor(np.array(nm), device=self.device)
        d_t = torch.as_tensor(d, device=self.device, dtype=torch.float)

        current_q = self.policy_net(s_t).gather(1, a_t).squeeze()
        
        with torch.no_grad():
            # Double DQN: Policy picks best action, Target evaluates value
            next_policy_q = self.policy_net(ns_t)
            next_policy_q[~nm_t] = -1e9
            next_actions = next_policy_q.argmax(dim=1, keepdim=True)
            
            next_target_q = self.target_net(ns_t).gather(1, next_actions).squeeze()
            expected_q = r_t + (GAMMA * next_target_q * (1 - d_t))

        loss = F.smooth_l1_loss(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()

def evaluate(agent):
    eval_env = FruitBoxEnv()
    state = eval_env.reset(seed=42)
    done = False
    while not done:
        mask = eval_env.get_valid_mask()
        if not any(mask): break
        action = agent.select_action(state, mask, 0, greedy=True)
        state, _, done, _ = eval_env.step(action)
    return eval_env.score

# --- 4. Main Loop ---
def main():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Score', 'Avg100', 'Epsilon', 'Loss', 'Greedy_Eval'])

    pygame.init()
    screen = pygame.display.set_mode((COLS*BLOCK_SIZE + 100, ROWS*BLOCK_SIZE + 100))
    env = FruitBoxEnv()
    agent = Agent(env.action_size)
    
    score_window = collections.deque(maxlen=100)
    
    for episode in range(1, 10001):
        state = env.reset()
        mask = env.get_valid_mask()
        done = False
        ep_losses = []

        while not done:
            epsilon = max(EPS_END, EPS_START - agent.steps / EPS_DECAY)
            action = agent.select_action(state, mask, epsilon)
            next_state, reward, done, next_mask = env.step(action)
            
            agent.memory.append((state, action, reward, next_state, next_mask, done))
            state, mask = next_state, next_mask
            agent.steps += 1
            
            loss = agent.train_step()
            if loss: ep_losses.append(loss)

        score_window.append(env.score)
        avg100 = np.mean(score_window)
        avg_ep_loss = np.mean(ep_losses) if ep_losses else 0
        greedy_eval = evaluate(agent)

        with open(CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, env.score, round(avg100, 2), round(epsilon, 3), round(avg_ep_loss, 4), greedy_eval])

        if episode % LOG_EVERY == 0:
            print(f"Episode {episode} | score={env.score} | avg100={avg100:.2f} | "
                  f"epsilon={epsilon:.3f} | replay={len(agent.memory)} | "
                  f"loss={avg_ep_loss:.4f} | greedy_eval={greedy_eval:.2f}")

        if episode % RENDER_EVERY == 0:
            screen.fill((30,30,30))
            for r in range(ROWS):
                for c in range(COLS):
                    if env.grid[r,c] > 0:
                        pygame.draw.rect(screen, (100, 100, 250), (c*30+50, r*30+50, 28, 28))
            pygame.display.flip()
            pygame.event.pump()

if __name__ == "__main__":
    main()
