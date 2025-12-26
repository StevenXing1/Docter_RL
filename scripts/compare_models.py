"""
Model comparison utility
Compares performance of multiple trained models
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def load_checkpoint_info(checkpoint_path):
    """Extract information from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        info = {
            "path": checkpoint_path,
            "episode": checkpoint.get("episode", "N/A"),
            "reward": checkpoint.get("reward", None),
        }

        # Get training scores if available
        if "scores" in checkpoint:
            scores = checkpoint["scores"]
            if len(scores) > 0:
                info["final_mean_reward"] = (
                    np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                )
                info["final_std_reward"] = (
                    np.std(scores[-100:]) if len(scores) >= 100 else np.std(scores)
                )
                info["max_reward"] = np.max(scores)
                info["min_reward"] = np.min(scores)
                info["total_episodes"] = len(scores)

        # Get model config if available
        if "config" in checkpoint:
            config = checkpoint["config"]
            info["model_type"] = config.get("model", {}).get("type", "Unknown")
            info["learning_rate"] = config.get("training", {}).get(
                "learning_rate", "N/A"
            )
            info["gamma"] = config.get("training", {}).get("gamma", "N/A")

        return info
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None


def compare_models(model_paths, output_dir=None):
    """Compare multiple models and generate report"""

    print("\n" + "=" * 80)
    print("MODEL COMPARISON REPORT")
    print("=" * 80)

    results = []

    for path in model_paths:
        print(f"\nAnalyzing: {path}")
        info = load_checkpoint_info(path)
        if info:
            results.append(info)

    if not results:
        print("No valid models found!")
        return

    # Create comparison table
    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(df.to_string(index=False))

    # Statistical comparison
    if "final_mean_reward" in df.columns:
        print("\n" + "=" * 80)
        print("PERFORMANCE RANKING (by Final Mean Reward)")
        print("=" * 80)

        ranked = df.sort_values("final_mean_reward", ascending=False)
        for idx, row in ranked.iterrows():
            model_name = Path(row["path"]).parent.name + "/" + Path(row["path"]).name
            print(f"{idx+1}. {model_name}")
            print(
                f"   Mean Reward: {row['final_mean_reward']:.2f} ± {row['final_std_reward']:.2f}"
            )
            print(f"   Max Reward: {row['max_reward']:.2f}")
            print()

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save CSV
        csv_path = output_path / "model_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved comparison table to: {csv_path}")

        # Generate plots if scores available
        plot_comparison(results, output_path)


def plot_comparison(results, output_dir):
    """Generate comparison plots"""

    # Filter results with scores
    results_with_scores = [r for r in results if "final_mean_reward" in r]

    if not results_with_scores:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Model names
    model_names = [Path(r["path"]).stem for r in results_with_scores]

    # Plot 1: Mean rewards with error bars
    ax = axes[0]
    means = [r["final_mean_reward"] for r in results_with_scores]
    stds = [r["final_std_reward"] for r in results_with_scores]

    x = np.arange(len(model_names))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Reward (Last 100 Episodes)")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 2: Best vs Average performance
    ax = axes[1]
    max_rewards = [r["max_reward"] for r in results_with_scores]

    x = np.arange(len(model_names))
    width = 0.35
    ax.bar(x - width / 2, means, width, label="Mean", alpha=0.7, edgecolor="black")
    ax.bar(x + width / 2, max_rewards, width, label="Max", alpha=0.7, edgecolor="black")

    ax.set_xlabel("Model")
    ax.set_ylabel("Reward")
    ax.set_title("Mean vs Maximum Reward")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: Training episodes
    ax = axes[2]
    episodes = [r.get("total_episodes", 0) for r in results_with_scores]

    bars = ax.bar(model_names, episodes, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Model")
    ax.set_ylabel("Training Episodes")
    ax.set_title("Training Duration")
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    # Color bars by performance
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.tight_layout()

    # Save
    plot_path = output_dir / "model_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved comparison plot to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare multiple trained models")
    parser.add_argument("models", nargs="+", help="Paths to model checkpoints")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/comparison",
        help="Output directory for comparison results",
    )

    args = parser.parse_args()

    # Validate paths
    valid_paths = []
    for path in args.models:
        p = Path(path)
        if p.exists():
            valid_paths.append(path)
        else:
            print(f"⚠ Warning: File not found: {path}")

    if not valid_paths:
        print("Error: No valid model paths provided!")
        return

    print(f"\nComparing {len(valid_paths)} models...")
    compare_models(valid_paths, args.output)


if __name__ == "__main__":
    main()
