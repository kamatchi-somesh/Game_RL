import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.coin_collector import CoinCollectorEnv
from agents.dqn import DQNAgent
import torch

def train():
    # Initialize environment
    env = CoinCollectorEnv(render_mode="human")
    env.reset()
    
    # Initialize agent (force CPU for compatibility)
    device = torch.device("cpu")  # Start with CPU for stability
    agent = DQNAgent(
        state_size=env.observation_space,
        action_size=env.action_space,
        device=device
    )
    
    # Training parameters
    batch_size = 32
    min_buffer_size = 1000
    max_episodes = 100
    
    print(f"Starting training on {device}...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    for episode in range(max_episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Handle gymnasium-style returns
            state = state[0]
            
        total_reward = 0
        done = False
        step_count = 0
        
        while not done:
            # Get action
            action = agent.act(state)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train if enough samples
            if len(agent.memory) > min_buffer_size:
                loss = agent.train(batch_size)
            else:
                loss = None
            
            total_reward += reward
            state = next_state
            step_count += 1
            
            # Early termination if stuck
            if step_count > 1000:
                print(f"Episode {episode} too long, terminating")
                break
        
        # Logging
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward {total_reward:.1f}, Steps {step_count}")
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/dqn_ep{episode}.pth")
    
    # Final save
    agent.save("models/dqn_final.pth")
    print("Training completed")

if __name__ == "__main__":
    train()