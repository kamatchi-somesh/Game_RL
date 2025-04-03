import pygame
from envs.coin_collector import CoinCollectorEnv
from agents.dqn import DQNAgent
import torch

def play_human():
    env = CoinCollectorEnv(render_mode="human")
    agent = DQNAgent(env.observation_space, env.action_space)
    agent.load("models/dqn_final.pth")
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        human_action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: human_action = 0
                elif event.key == pygame.K_DOWN: human_action = 1
                elif event.key == pygame.K_LEFT: human_action = 2
                elif event.key == pygame.K_RIGHT: human_action = 3
        
        if human_action is not None:
            state = env._get_state()
            agent_action = agent.act(state, epsilon=0.01)
            env.step(agent_action, human_action)
            env.render()
        
        clock.tick(10)
    
    env.close()

if __name__ == "__main__":
    play_human()