import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir=None):
        self.log_dir = log_dir or os.path.join("runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        self.episode_rewards = []
        self.losses = []
        
    def log_episode(self, episode, reward, loss=None):
        """Log metrics for an episode"""
        self.episode_rewards.append(reward)
        self.writer.add_scalar("Reward/episode", reward, episode)
        self.writer.add_scalar("Reward/mean_100", np.mean(self.episode_rewards[-100:]), episode)
        
        if loss is not None:
            self.losses.append(loss)
            self.writer.add_scalar("Loss/train", loss, episode)
    
    def save_plots(self):
        """Save performance plots to data/"""
        os.makedirs("data", exist_ok=True)
        
        # Reward plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards)
        plt.plot(np.convolve(self.episode_rewards, np.ones(100)/100, mode='valid'))
        plt.title("Training Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig(os.path.join("data", "rewards.png"))
        plt.close()
        
        # Loss plot
        if self.losses:
            plt.figure(figsize=(10, 5))
            plt.plot(self.losses)
            plt.title("Training Losses")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.savefig(os.path.join("data", "losses.png"))
            plt.close()
    
    def close(self):
        self.writer.close()
        self.save_plots()