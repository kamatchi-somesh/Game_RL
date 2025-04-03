import random
import numpy as np
from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, capacity=10000, device=None):
        self.buffer = deque(maxlen=capacity)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32)
        ))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
            
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device),
            torch.as_tensor(np.array(actions), dtype=torch.long, device=self.device),
            torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device),
            torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device),
            torch.as_tensor(np.array(dones), dtype=torch.float32, device=self.device)
        )

    def __len__(self):
        return len(self.buffer)