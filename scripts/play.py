import pygame
from envs.coin_collector import CoinCollectorEnv
from agents.dqn import DQNAgent
import torch

def load_agent(model_path, state_size, action_size, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load a trained DQN agent with automatic CUDA detection"""
    # Initialize agent
    agent = DQNAgent(state_size, action_size, device)
    
    # Load model weights with proper device mapping
    if device == "cuda":
        agent.q_net.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    else:
        agent.q_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Sync target network and set evaluation mode
    agent.update_target_net()
    agent.q_net.eval()
    agent.target_net.eval()
    
    return agent

def agent_vs_agent():
    env = CoinCollectorEnv(grid_size=10)
    agent1 = load_agent("models/dqn_model.pth", env.state_size, 4)
    agent2 = load_agent("models/dqn_model.pth", env.state_size, 4)  # or a different agent

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Agent 1 (DQN)
        state1 = env._get_state()
        action1 = agent1.act(state1, epsilon=0.01)  # Low epsilon for evaluation

        # Agent 2 (DQN or scripted)
        state2 = env._get_state()
        action2 = agent2.act(state2, epsilon=0.01)

        # Step environment
        env.step(action1, action2)
        env.render()
        clock.tick(5)  # Control speed

    pygame.quit()

if __name__ == "__main__":
    agent_vs_agent()  # Or human_vs_agent()