import argparse
import torch

from pursuit_evasion import PursuerPolicy, load_config
from train_pursuer import PursuerOnlyEnv
from train_pursuer_ppo import ActorCritic


def run_episode(model_path: str, use_ppo: bool = False) -> None:
    cfg = load_config()
    cfg['evader']['awareness_mode'] = 1
    env = PursuerOnlyEnv(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_ppo:
        model = ActorCritic(env.observation_space.shape[0])
    else:
        model = PursuerPolicy(env.observation_space.shape[0])
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            if use_ppo:
                mean, _ = model(obs_t)
            else:
                mean = model(obs_t)
            dist = torch.distributions.Normal(mean, torch.ones_like(mean))
            action = dist.mean
        obs, r, done, _, _ = env.step(action.cpu().numpy())
        total_reward += r

    print(f"Episode reward: {total_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single episode using a saved pursuer model"
    )
    parser.add_argument(
        "--model", type=str, default="pursuer_policy.pt", help="path to weight file"
    )
    parser.add_argument(
        "--ppo", action="store_true", help="load weights from PPO training"
    )
    args = parser.parse_args()

    run_episode(args.model, use_ppo=args.ppo)
