"""
Training script for RL Doctor Agent
Supports MLP, RNN, and LSTM policies with configurable hyperparameters
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from datetime import datetime
import gym

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import final_proj

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


class PolicyMLP(nn.Module):
    """Multi-Layer Perceptron Policy"""

    def __init__(self, s_size, a_size, h_size=128):
        super(PolicyMLP, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=1)

    def act(self, state, device):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class PolicyRNN(nn.Module):
    """Recurrent Neural Network Policy"""

    def __init__(self, s_size, a_size, h_size=128, num_layers=2, dropout=0.5):
        super(PolicyRNN, self).__init__()
        self.h_size = h_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            s_size,
            h_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Linear(h_size, a_size)

    def forward(self, x, hidden=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return torch.softmax(out, dim=1), hidden

    def act(self, state, device, hidden=None):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs, hidden = self.forward(state, hidden)
        probs = probs.cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action), hidden


class PolicyLSTM(nn.Module):
    """Long Short-Term Memory Policy"""

    def __init__(self, s_size, a_size, h_size=128, num_layers=2, dropout=0.3):
        super(PolicyLSTM, self).__init__()
        self.h_size = h_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            s_size,
            h_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Linear(h_size, a_size)

    def forward(self, x, hidden=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return torch.softmax(out, dim=1), hidden

    def act(self, state, device, hidden=None):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs, hidden = self.forward(state, hidden)
        probs = probs.cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action), hidden


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model(config, s_size, a_size, device):
    """Create policy model based on configuration"""
    model_type = config["model"]["type"].lower()
    arch = config["model"]["architecture"]

    if model_type == "mlp":
        h_size = arch.get("hidden_sizes", [128])[0]
        model = PolicyMLP(s_size, a_size, h_size)
    elif model_type == "rnn":
        h_size = arch.get("hidden_size", 128)
        num_layers = arch.get("num_layers", 2)
        dropout = arch.get("dropout", 0.5)
        model = PolicyRNN(s_size, a_size, h_size, num_layers, dropout)
    elif model_type == "lstm":
        h_size = arch.get("hidden_size", 128)
        num_layers = arch.get("num_layers", 2)
        dropout = arch.get("dropout", 0.3)
        model = PolicyLSTM(s_size, a_size, h_size, num_layers, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def reinforce(policy, optimizer, episode_rewards, episode_log_probs, gamma, device):
    """REINFORCE algorithm update"""
    # Compute returns
    returns = []
    G = 0
    for r in reversed(episode_rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns).to(device)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Compute loss
    policy_loss = []
    for log_prob, G in zip(episode_log_probs, returns):
        policy_loss.append(-log_prob * G)

    optimizer.zero_grad()
    loss = torch.cat(policy_loss).sum()
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)

    optimizer.step()

    return loss.item()


def train(config, model, env, device, writer=None):
    """Main training loop"""
    train_config = config["training"]
    log_config = config["logging"]
    ckpt_config = config["checkpoint"]

    optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])
    gamma = train_config["gamma"]
    num_episodes = train_config["episodes"]
    max_steps = train_config["max_steps"]

    best_reward = -float("inf")
    scores = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_rewards = []
        episode_log_probs = []
        hidden = None

        for step in range(max_steps):
            # Get action
            if isinstance(model, PolicyMLP):
                action, log_prob = model.act(state, device)
            else:
                action, log_prob, hidden = model.act(state, device, hidden)

            # Take step
            next_state, reward, done, _ = env.step(action)

            episode_rewards.append(reward)
            episode_log_probs.append(log_prob)

            state = next_state

            if done:
                break

        # Update policy
        episode_return = sum(episode_rewards)
        scores.append(episode_return)

        loss = reinforce(
            model, optimizer, episode_rewards, episode_log_probs, gamma, device
        )

        # Logging
        if writer and episode % 10 == 0:
            writer.add_scalar("Train/Episode_Reward", episode_return, episode)
            writer.add_scalar("Train/Loss", loss, episode)
            writer.add_scalar("Train/Episode_Length", len(episode_rewards), episode)

        # Print progress
        if episode % log_config["eval_frequency"] == 0:
            avg_score = np.mean(scores[-100:])
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Avg Score: {avg_score:.2f} | "
                f"Last Score: {episode_return:.2f} | "
                f"Loss: {loss:.4f}"
            )

            if writer:
                writer.add_scalar("Train/Avg_Reward_100", avg_score, episode)

        # Save checkpoints
        if episode % log_config["save_frequency"] == 0 or episode_return > best_reward:
            os.makedirs(ckpt_config["save_dir"], exist_ok=True)

            if episode_return > best_reward and ckpt_config["save_best"]:
                best_reward = episode_return
                save_path = os.path.join(ckpt_config["save_dir"], "best_model.pth")
                torch.save(
                    {
                        "episode": episode,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "reward": best_reward,
                    },
                    save_path,
                )
                print(f"✓ Saved best model with reward: {best_reward:.2f}")

            if ckpt_config["save_last"]:
                save_path = os.path.join(
                    ckpt_config["save_dir"], f"checkpoint_ep{episode}.pth"
                )
                torch.save(
                    {
                        "episode": episode,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "reward": episode_return,
                    },
                    save_path,
                )

    return scores


def main():
    parser = argparse.ArgumentParser(description="Train RL Doctor Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mlp_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "rnn", "lstm"],
        help="Model type (overrides config)",
    )
    parser.add_argument(
        "--episodes", type=int, help="Number of episodes (overrides config)"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device: cuda, cpu, or auto"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    if args.model:
        config["model"]["type"] = args.model
        args.config = f"configs/{args.model}_config.yaml"
        config = load_config(args.config)
    if args.episodes:
        config["training"]["episodes"] = args.episodes

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment
    env = gym.make(config["environment"]["name"])
    env.seed(args.seed)

    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    print(f"\nEnvironment: {config['environment']['name']}")
    print(f"State size: {s_size}")
    print(f"Action size: {a_size}")

    # Create model
    model = create_model(config, s_size, a_size, device)
    print(f"\nModel: {config['model']['type'].upper()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup logging
    writer = None
    if TENSORBOARD_AVAILABLE and config["logging"]["tensorboard"]:
        log_dir = config["logging"]["log_dir"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"{log_dir}_{timestamp}"
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")

    # Train
    print(f"\nStarting training for {config['training']['episodes']} episodes...")
    scores = train(config, model, env, device, writer)

    # Save final model
    os.makedirs(config["checkpoint"]["save_dir"], exist_ok=True)
    final_path = os.path.join(config["checkpoint"]["save_dir"], "final_model.pth")
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config, "scores": scores},
        final_path,
    )
    print(f"\n✓ Training complete! Final model saved to: {final_path}")

    if writer:
        writer.close()

    env.close()


if __name__ == "__main__":
    main()
