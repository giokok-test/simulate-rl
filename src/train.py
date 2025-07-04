import torch
import numpy as np
from environment import Simple3DEnv
from model import SimplePolicyNetwork


def train(num_episodes: int = 10):
    env = Simple3DEnv()
    policy = SimplePolicyNetwork()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            state_tensor = torch.from_numpy(state).float()
            action = policy(state_tensor).detach().numpy()
            next_state, reward, done, info = env.step(action)

            ep_reward += reward
            loss = -torch.tensor(reward, dtype=torch.float32)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
        print(f"Episode {episode}: reward {ep_reward:.2f}")


if __name__ == "__main__":
    train()
