# Project Architecture

## Overview

This document describes the architecture and design decisions of the Doctor RL project.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│  (CLI Scripts: train.py, evaluate.py, visualize.py)    │
└───────────────────┬─────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────┐
│              Configuration Layer                         │
│         (YAML configs, hyperparameters)                 │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────┴──────────┐
        │                      │
┌───────▼────────┐    ┌───────▼────────┐
│  RL Algorithms │    │  Logging/Utils │
│  (REINFORCE)   │    │  (TensorBoard) │
└───────┬────────┘    └────────────────┘
        │
┌───────▼─────────────────────────────────────────────────┐
│                 Policy Networks                          │
│          (MLP, RNN, LSTM architectures)                 │
└───────────────────┬─────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────┐
│              Gym Environment                             │
│         (DocterEnv - Blood Pressure Sim)                │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐      │
│  │   State    │  │   Action   │  │   Reward   │      │
│  │ (BP data)  │  │(intervene?)│  │ (BP score) │      │
│  └────────────┘  └────────────┘  └────────────┘      │
└──────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Environment Layer (`final_proj/envs/`)

**DocterEnv**: Custom Gym environment simulating blood pressure dynamics
- Inherits from `gym.Env`
- Implements standard RL interface: `reset()`, `step()`, `render()`
- State space: Blood pressure waveform features
- Action space: Discrete (intervene or not)
- Reward: Based on maintaining healthy BP range

**Key Files:**
- `docterEnv.py`: Main environment implementation
- `docter.py`: Game logic and physics
- `base/pygamewrapper.py`: Pygame integration

### 2. Policy Networks (`scripts/train.py`)

Three policy architectures implemented:

**PolicyMLP**: Feed-forward neural network
- Input → FC128 → ReLU → FC128 → ReLU → FC(actions) → Softmax
- Fast, suitable for Markovian states
- ~20K parameters

**PolicyRNN**: Recurrent neural network
- Input → RNN(128, 2 layers) → FC(actions) → Softmax
- Maintains hidden state across timesteps
- Handles temporal dependencies

**PolicyLSTM**: Long Short-Term Memory
- Input → LSTM(128, 2 layers) → FC(actions) → Softmax
- Cell state + hidden state
- Best for long-term dependencies

### 3. Training Algorithm

**REINFORCE (Monte Carlo Policy Gradient)**
```
For each episode:
  1. Collect trajectory: (s₀, a₀, r₀), ..., (sₜ, aₜ, rₜ)
  2. Compute returns: Gₜ = Σ γⁱrₜ₊ᵢ
  3. Normalize returns: Ĝₜ = (Gₜ - μ) / σ
  4. Update policy: ∇θ J(θ) = Σ Ĝₜ ∇ log π(aₜ|sₜ)
  5. Apply gradient clipping
  6. Optimizer step
```

### 4. Configuration System (`configs/`)

YAML-based configuration with sections:
- **model**: Architecture type and hyperparameters
- **training**: Algorithm settings (LR, gamma, episodes)
- **environment**: Env creation parameters
- **logging**: TensorBoard and checkpoint settings
- **checkpoint**: Model saving strategy

### 5. Logging Infrastructure

**ExperimentLogger** (`final_proj/utils/logger.py`)
- JSON-based metric tracking
- Automatic experiment naming
- Summary statistics
- TensorBoard integration (optional)

**MetricsTracker**
- Rolling window statistics
- Episode rewards, lengths, losses
- Progress reporting

### 6. Evaluation Framework (`scripts/evaluate.py`)

- Model loading and inference
- Multi-episode evaluation
- Statistical analysis
- Results visualization

## Data Flow

### Training Pipeline

```
Config File → Load Config → Create Environment
                ↓
        Create Policy Model
                ↓
        Initialize Optimizer
                ↓
        Training Loop:
          ├─ Reset Environment
          ├─ Collect Episode (states, actions, rewards)
          ├─ Compute Returns
          ├─ Policy Gradient Update
          ├─ Log Metrics
          └─ Save Checkpoints
                ↓
        Save Final Model
```

### Evaluation Pipeline

```
Checkpoint File → Load Model → Create Environment
                ↓
        Evaluation Loop:
          ├─ Reset Environment
          ├─ Run Episode (greedy policy)
          ├─ Collect Statistics
          └─ Render (optional)
                ↓
        Compute Metrics (mean, std, etc.)
                ↓
        Save Results / Generate Plots
```

## Design Decisions

### Why REINFORCE?

- Simple, effective policy gradient method
- Works well for episodic tasks
- Good baseline before trying advanced methods (PPO, A3C)
- Clear gradient computation and implementation

### Why Multiple Architectures?

- **MLP**: Baseline, fast training
- **RNN**: Compare temporal modeling
- **LSTM**: Handle long-term dependencies in BP dynamics

### Configuration Management

- YAML format: Human-readable, easy to version control
- Separate configs per model: Isolate hyperparameters
- Override via CLI: Flexible experimentation

### Modular Scripts

Separate scripts for different tasks:
- `train.py`: Focus on training logic
- `evaluate.py`: Independent evaluation
- `visualize.py`: Plotting and analysis

Benefits:
- Easier debugging
- Reusable components
- Clean interfaces

## Testing Strategy

### Unit Tests (`tests/`)

1. **Environment Tests** (`test_environment.py`)
   - Interface compliance (Gym API)
   - State/action space properties
   - Deterministic seeding
   - Episode execution

2. **Model Tests** (`test_models.py`)
   - Architecture creation
   - Forward pass
   - Action sampling
   - Training step
   - Save/load checkpoints

### Integration Tests

- Full training pipeline (few episodes)
- Evaluation pipeline
- Config loading and validation

## Performance Considerations

### Computational Efficiency

- Gradient clipping prevents exploding gradients
- Batch size = 1 (online learning) for stability
- Episode-level updates (Monte Carlo)

### Memory Management

- Clear episode buffers after update
- No replay buffer needed (on-policy)
- Model checkpointing for large experiments

### Scalability

- TensorBoard for experiment tracking
- Config-based hyperparameter sweeps
- Parallel evaluation possible

## Future Enhancements

### Algorithmic Improvements

- [ ] PPO (Proximal Policy Optimization)
- [ ] A2C/A3C (Actor-Critic methods)
- [ ] Baseline function (value network)
- [ ] Experience replay

### Environment Extensions

- [ ] Multi-patient scenarios
- [ ] Continuous action space
- [ ] Partial observability
- [ ] Stochastic dynamics

### Engineering

- [ ] Distributed training
- [ ] Hyperparameter optimization (Optuna)
- [ ] Model serving API
- [ ] Web dashboard

## References

- Sutton & Barto: Reinforcement Learning: An Introduction
- OpenAI Spinning Up in Deep RL
- PyTorch Documentation
- OpenAI Gym Documentation
