"""
Evaluation script for trained RL Doctor Agent models
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import numpy as np
import torch
import gym
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import final_proj

# Import models from train.py
from train import PolicyMLP, PolicyRNN, PolicyLSTM


def load_model(model_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get config if available
    config = checkpoint.get('config', None)
    
    return checkpoint, config


def evaluate(model, env, num_episodes=100, render=False, device='cpu'):
    """Evaluate model performance"""
    stats = defaultdict(list)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        hidden = None
        
        done = False
        while not done:
            with torch.no_grad():
                if isinstance(model, PolicyMLP):
                    action, _ = model.act(state, device)
                else:
                    action, _, hidden = model.act(state, device, hidden)
            
            next_state, reward, done, info = env.step(action)
            
            if render:
                env.render()
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
        
        stats['rewards'].append(episode_reward)
        stats['steps'].append(episode_steps)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Reward: {episode_reward:.2f}, Steps: {episode_steps}")
    
    return stats


def print_statistics(stats):
    """Print evaluation statistics"""
    rewards = np.array(stats['rewards'])
    steps = np.array(stats['steps'])
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Episodes:         {len(rewards)}")
    print(f"\nReward Statistics:")
    print(f"  Mean:           {rewards.mean():.2f}")
    print(f"  Std:            {rewards.std():.2f}")
    print(f"  Min:            {rewards.min():.2f}")
    print(f"  Max:            {rewards.max():.2f}")
    print(f"  Median:         {np.median(rewards):.2f}")
    print(f"\nEpisode Length:")
    print(f"  Mean:           {steps.mean():.2f}")
    print(f"  Std:            {steps.std():.2f}")
    print(f"  Min:            {steps.min()}")
    print(f"  Max:            {steps.max()}")
    print("="*60)


def save_results(stats, output_path):
    """Save evaluation results"""
    np.savez(output_path,
             rewards=stats['rewards'],
             steps=stats['steps'],
             mean_reward=np.mean(stats['rewards']),
             std_reward=np.std(stats['rewards']))
    print(f"\n✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate RL Doctor Agent')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model-type', type=str, choices=['mlp', 'rnn', 'lstm'],
                       help='Model architecture type')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--env', type=str, default='final_proj/RLDocter_v0',
                       help='Environment name')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output', type=str, default='results/evaluation.npz',
                       help='Output path for results')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading model from: {args.model_path}")
    checkpoint, config = load_model(args.model_path, device)
    
    # Create environment
    env = gym.make(args.env)
    env.seed(args.seed)
    
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n
    
    # Determine model type
    if args.model_type:
        model_type = args.model_type
    elif config:
        model_type = config['model']['type']
    else:
        # Try to infer from path
        if 'mlp' in args.model_path.lower():
            model_type = 'mlp'
        elif 'rnn' in args.model_path.lower():
            model_type = 'rnn'
        elif 'lstm' in args.model_path.lower():
            model_type = 'lstm'
        else:
            raise ValueError("Cannot determine model type. Please specify --model-type")
    
    print(f"Model type: {model_type.upper()}")
    
    # Create model
    if model_type == 'mlp':
        model = PolicyMLP(s_size, a_size)
    elif model_type == 'rnn':
        model = PolicyRNN(s_size, a_size)
    elif model_type == 'lstm':
        model = PolicyLSTM(s_size, a_size)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"\nEvaluating for {args.episodes} episodes...")
    
    # Evaluate
    stats = evaluate(model, env, args.episodes, args.render, device)
    
    # Print results
    print_statistics(stats)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_results(stats, args.output)
    
    env.close()


if __name__ == '__main__':
    main()
