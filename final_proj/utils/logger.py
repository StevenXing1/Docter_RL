"""
Logging utilities for RL training
Provides structured logging, metrics tracking, and experiment management
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np


class ExperimentLogger:
    """Logger for RL experiments with metric tracking"""
    
    def __init__(self, log_dir, experiment_name=None):
        """
        Initialize experiment logger
        
        Args:
            log_dir: Base directory for logs
            experiment_name: Name of experiment (auto-generated if None)
        """
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        
        # Create subdirectories
        (self.log_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.log_dir / 'plots').mkdir(exist_ok=True)
        
        print(f"Logging to: {self.log_dir}")
    
    def log_metric(self, name, value, step=None):
        """Log a metric value"""
        entry = {
            'value': float(value),
            'timestamp': time.time() - self.start_time
        }
        if step is not None:
            entry['step'] = step
        
        self.metrics[name].append(entry)
    
    def log_metrics(self, metrics_dict, step=None):
        """Log multiple metrics at once"""
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step)
    
    def log_config(self, config):
        """Save configuration to JSON file"""
        config_path = self.log_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Saved config to: {config_path}")
    
    def log_hyperparameters(self, hparams):
        """Save hyperparameters"""
        hparams_path = self.log_dir / 'hyperparameters.json'
        with open(hparams_path, 'w') as f:
            json.dump(hparams, f, indent=2)
    
    def save_metrics(self):
        """Save all metrics to JSON file"""
        metrics_path = self.log_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
        print(f"✓ Saved metrics to: {metrics_path}")
    
    def get_metric_stats(self, name, window=100):
        """Get statistics for a metric"""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return None
        
        values = [m['value'] for m in self.metrics[name]]
        recent = values[-window:] if len(values) >= window else values
        
        return {
            'mean': np.mean(recent),
            'std': np.std(recent),
            'min': np.min(recent),
            'max': np.max(recent),
            'last': values[-1],
            'count': len(values)
        }
    
    def print_summary(self):
        """Print summary of logged metrics"""
        print("\n" + "="*60)
        print(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        print("="*60)
        
        for name in sorted(self.metrics.keys()):
            stats = self.get_metric_stats(name)
            if stats:
                print(f"\n{name}:")
                print(f"  Last:  {stats['last']:.4f}")
                print(f"  Mean:  {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"  Count: {stats['count']}")
        
        elapsed = time.time() - self.start_time
        print(f"\nTotal Time: {elapsed/60:.2f} minutes")
        print("="*60)


class MetricsTracker:
    """Simple metrics tracker for episode statistics"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics"""
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
    
    def add_episode(self, reward, length):
        """Add episode data"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
    
    def add_loss(self, loss):
        """Add training loss"""
        self.losses.append(loss)
    
    def get_stats(self):
        """Get current statistics"""
        recent_rewards = self.episode_rewards[-self.window_size:]
        recent_lengths = self.episode_lengths[-self.window_size:]
        recent_losses = self.losses[-self.window_size:]
        
        stats = {}
        
        if recent_rewards:
            stats['reward_mean'] = np.mean(recent_rewards)
            stats['reward_std'] = np.std(recent_rewards)
            stats['reward_max'] = np.max(recent_rewards)
            stats['reward_min'] = np.min(recent_rewards)
        
        if recent_lengths:
            stats['length_mean'] = np.mean(recent_lengths)
        
        if recent_losses:
            stats['loss_mean'] = np.mean(recent_losses)
        
        return stats
    
    def print_progress(self, episode):
        """Print training progress"""
        stats = self.get_stats()
        
        msg = f"Episode {episode}"
        if 'reward_mean' in stats:
            msg += f" | Reward: {stats['reward_mean']:.2f} ± {stats['reward_std']:.2f}"
        if 'length_mean' in stats:
            msg += f" | Length: {stats['length_mean']:.1f}"
        if 'loss_mean' in stats:
            msg += f" | Loss: {stats['loss_mean']:.4f}"
        
        print(msg)


def create_logger(log_dir, experiment_name=None, use_tensorboard=True):
    """
    Factory function to create appropriate logger
    
    Args:
        log_dir: Base directory for logs
        experiment_name: Name of experiment
        use_tensorboard: Whether to use TensorBoard (if available)
    
    Returns:
        Logger instance
    """
    logger = ExperimentLogger(log_dir, experiment_name)
    
    # Try to add TensorBoard if available
    if use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(str(logger.log_dir / 'tensorboard'))
            logger.tb_writer = tb_writer
            print("✓ TensorBoard logging enabled")
        except ImportError:
            logger.tb_writer = None
            print("⚠ TensorBoard not available")
    else:
        logger.tb_writer = None
    
    return logger


if __name__ == '__main__':
    # Example usage
    logger = ExperimentLogger('logs', 'test_experiment')
    
    # Log some metrics
    for i in range(100):
        logger.log_metric('reward', np.random.randn() + i * 0.1, step=i)
        logger.log_metric('loss', np.random.rand() * 0.1, step=i)
    
    # Print summary
    logger.print_summary()
    
    # Save metrics
    logger.save_metrics()
