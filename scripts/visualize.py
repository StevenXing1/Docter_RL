"""
Visualization utilities for training results and agent performance
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch


def plot_training_curves(scores, window=100, save_path=None):
    """Plot training reward curves with moving average"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Raw scores
    ax1.plot(scores, alpha=0.3, label='Raw Score')
    
    # Moving average
    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(scores)), moving_avg, 
                linewidth=2, label=f'{window}-Episode Moving Average')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribution of scores
    ax2.hist(scores, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(scores), color='r', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(scores):.2f}')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Reward Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training plot to: {save_path}")
    else:
        plt.show()


def plot_evaluation_results(results_path, save_path=None):
    """Plot evaluation results"""
    data = np.load(results_path)
    rewards = data['rewards']
    steps = data['steps']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Reward over episodes
    ax = axes[0, 0]
    ax.plot(rewards, marker='o', markersize=3, alpha=0.6)
    ax.axhline(np.mean(rewards), color='r', linestyle='--', 
              label=f'Mean: {np.mean(rewards):.2f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Evaluation Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Reward distribution
    ax = axes[0, 1]
    ax.hist(rewards, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(rewards), color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Total Reward')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Reward Distribution (μ={np.mean(rewards):.2f}, σ={np.std(rewards):.2f})')
    ax.grid(True, alpha=0.3)
    
    # Episode length over episodes
    ax = axes[1, 0]
    ax.plot(steps, marker='o', markersize=3, alpha=0.6, color='green')
    ax.axhline(np.mean(steps), color='r', linestyle='--',
              label=f'Mean: {np.mean(steps):.1f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length')
    ax.set_title('Episode Lengths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box plot summary
    ax = axes[1, 1]
    bp = ax.boxplot([rewards, steps], labels=['Rewards', 'Steps'],
                    patch_artist=True, showmeans=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel('Value')
    ax.set_title('Statistical Summary')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved evaluation plot to: {save_path}")
    else:
        plt.show()


def compare_models(model_paths, labels=None, save_path=None):
    """Compare multiple trained models"""
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(model_paths))]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    all_scores = []
    for path, label in zip(model_paths, labels):
        checkpoint = torch.load(path, map_location='cpu')
        scores = checkpoint.get('scores', [])
        all_scores.append(scores)
    
    # Training curves
    ax = axes[0]
    for scores, label in zip(all_scores, labels):
        if len(scores) > 0:
            window = min(100, len(scores))
            moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(scores)), moving_avg, label=label, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (Moving Average)')
    ax.set_title('Training Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final performance comparison
    ax = axes[1]
    final_rewards = [np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores) 
                    for scores in all_scores]
    bars = ax.bar(labels, final_rewards, alpha=0.7, edgecolor='black')
    
    # Color bars by performance
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_ylabel('Average Reward (Last 100 Episodes)')
    ax.set_title('Final Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize RL results')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['training', 'evaluation', 'compare'],
                       help='Visualization mode')
    parser.add_argument('--input', type=str, nargs='+', required=True,
                       help='Input file(s) - model checkpoint or results file')
    parser.add_argument('--labels', type=str, nargs='+',
                       help='Labels for comparison mode')
    parser.add_argument('--output', type=str,
                       help='Output path for saving plot')
    parser.add_argument('--window', type=int, default=100,
                       help='Window size for moving average')
    
    args = parser.parse_args()
    
    if args.mode == 'training':
        # Load training checkpoint
        checkpoint = torch.load(args.input[0], map_location='cpu')
        scores = checkpoint.get('scores', [])
        
        if len(scores) == 0:
            print("No training scores found in checkpoint!")
            return
        
        plot_training_curves(scores, args.window, args.output)
    
    elif args.mode == 'evaluation':
        plot_evaluation_results(args.input[0], args.output)
    
    elif args.mode == 'compare':
        compare_models(args.input, args.labels, args.output)


if __name__ == '__main__':
    main()
