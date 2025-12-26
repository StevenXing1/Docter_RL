# Quick Start Guide

This guide will help you get started with training and evaluating the Doctor RL agent.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Docter_RL.git
cd Docter_RL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Training Your First Model

### 1. Train MLP Policy

```bash
python scripts/train.py --config configs/mlp_config.yaml
```

This will:
- Train a Multi-Layer Perceptron policy
- Save checkpoints every 100 episodes
- Log training metrics to TensorBoard
- Save the best model based on reward

### 2. Monitor Training

Open TensorBoard to monitor training progress:

```bash
tensorboard --logdir=logs/
```

Navigate to `http://localhost:6006` in your browser.

### 3. Evaluate Trained Model

After training completes, evaluate the model:

```bash
python scripts/evaluate.py --model-path models/mlp/best_model.pth --model-type mlp --episodes 100
```

### 4. Visualize Results

Generate training curve plots:

```bash
python scripts/visualize.py --mode training --input models/mlp/final_model.pth --output figures/training_curves.png
```

Generate evaluation plots:

```bash
python scripts/visualize.py --mode evaluation --input results/evaluation.npz --output figures/eval_results.png
```

## Training Other Models

### Train RNN Policy

```bash
python scripts/train.py --config configs/rnn_config.yaml
```

### Train LSTM Policy

```bash
python scripts/train.py --config configs/lstm_config.yaml
```

## Customizing Training

### Override Config Parameters

```bash
# Change number of episodes
python scripts/train.py --config configs/mlp_config.yaml --episodes 2000

# Use GPU
python scripts/train.py --config configs/mlp_config.yaml --device cuda
```

### Modify Configuration Files

Edit `configs/mlp_config.yaml` to change hyperparameters:

```yaml
training:
  episodes: 1000        # Number of training episodes
  learning_rate: 0.001  # Optimizer learning rate
  gamma: 0.99          # Discount factor
  max_steps: 300       # Max steps per episode
```

## Running Tests

Verify installation by running tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_environment.py -v

# Check coverage
pytest tests/ --cov=final_proj --cov-report=html
```

## Common Issues

### pygame display error

If you encounter SDL/pygame display errors:

```python
# The environment automatically sets display mode to dummy
# when display=False in the config
```

### CUDA out of memory

Reduce model size or use CPU:

```bash
python scripts/train.py --config configs/mlp_config.yaml --device cpu
```

## Next Steps

- Experiment with different hyperparameters
- Try custom reward functions in the environment
- Implement new policy architectures
- Compare model performances

For more details, see the full [README.md](README.md).
