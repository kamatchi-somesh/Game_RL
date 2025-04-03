import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from utils.replay_buffer import ReplayBuffer
class DQNAgent:
    def __init__(self, state_size, action_size, device="cpu"):
        self.device = torch.device(device)
        print(f"Initializing DQNAgent on {self.device}")
        
        # Simple network that matches demo's observations
        self.q_net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(self.device)
        
        self.target_net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.001)
        self.memory = ReplayBuffer(device=self.device)
        self.update_target_net()

    def act(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(self.q_net[-1].out_features)
        
        # Simple numpy to tensor conversion
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(state_tensor).argmax().item()

    def train(self, batch_size):
        batch = self.memory.sample(batch_size)
        if batch is None:
            return None
            
        states, actions, rewards, next_states, dones = batch
        
        # Current Q values
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + 0.99 * next_q * (1 - dones)
        
        # MSE loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target_net()