# Doctor Agent for Hypotension Management using Reinforcement Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Overview

This project develops an AI-driven medical decision support system for managing hypotension (low blood pressure) using state-of-the-art Deep Reinforcement Learning techniques. The agent learns optimal intervention strategies through interaction with a custom Pygame-based medical simulation environment.

**Key Features:**
- 🏥 Custom medical simulation environment modeling blood pressure dynamics
- 🤖 Multiple RL architectures: MLP, RNN, and LSTM-based policies
- 📊 Comprehensive logging and visualization tools
- 🧪 Evaluation framework with medical metrics
- 📦 Modular and extensible codebase

## 🎯 Background

Hypotension (systolic BP < 90 mmHg) can lead to organ damage and requires timely intervention. This project provides:
- **Real-time decision support** for medical professionals
- **Policy learning** from simulated patient responses
- **Interpretable interventions** based on blood pressure waveforms

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Docter_RL.git
cd Docter_RL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
pip install -r requirements.txt
```

### Basic Usage

```python
import gym
import final_proj

# Create environment
env = gym.make('final_proj/RLDocter_v0')

# Run episode
state = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    if done:
        break
```

### Training

```bash
# Train MLP policy
python scripts/train.py --model mlp --episodes 1000

# Train LSTM policy
python scripts/train.py --model lstm --episodes 1000 --config configs/lstm_config.yaml
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py --model-path models/mlp_model.pth --episodes 100
```

## 📁 Project Structure

```
Docter_RL/
├── final_proj/              # Main package
│   ├── envs/               # Custom Gym environments
│   │   └── docterEnv.py    # Blood pressure environment
│   ├── base/               # Base wrappers
│   │   ├── pygamewrapper.py
│   │   └── doomwrapper.py
│   ├── utils/              # Utility functions
│   └── docter.py           # Game logic
├── scripts/                # Training and evaluation scripts
│   ├── train.py           # Main training script
│   ├── evaluate.py        # Model evaluation
│   └── visualize.py       # Results visualization
├── configs/                # Configuration files
│   ├── mlp_config.yaml
│   ├── rnn_config.yaml
│   └── lstm_config.yaml
├── models/                 # Saved model checkpoints
├── logs/                   # Training logs and tensorboard
├── tests/                  # Unit tests
├── final_project.ipynb     # Demo notebook
├── requirements.txt        # Python dependencies
└── setup.py               # Package setup
```

## 🧠 Models

### Supported Architectures

1. **MLP (Multi-Layer Perceptron)**
   - Fast training, good baseline
   - Best for Markovian state representations

2. **RNN (Recurrent Neural Network)**
   - Handles temporal dependencies
   - Good for sequential medical data

3. **LSTM (Long Short-Term Memory)**
   - Best for long-term dependencies
   - Recommended for medical time series

### Model Performance

| Model | Mean Reward | Training Time | Convergence |
|-------|------------|---------------|-------------|
| MLP   | TBD        | ~10 min      | 500 episodes |
| RNN   | TBD        | ~15 min      | 800 episodes |
| LSTM  | TBD        | ~20 min      | 1000 episodes |

## 📊 Environment Details

**State Space:** Blood pressure waveform features (continuous)
- Current BP value
- BP trend (derivative)
- Time since intervention
- Patient state features

**Action Space:** Discrete (2 actions)
- `0`: No intervention
- `1`: Apply medication/intervention

**Reward Function:**
- Positive reward for maintaining BP in healthy range (80-120 mmHg)
- Negative reward for hypotensive states (<80 mmHg)
- Penalty for excessive interventions

## 🔬 Research & Development

### Running Experiments

```bash
# Hyperparameter sweep
python scripts/train.py --sweep --config configs/sweep.yaml

# Compare models
python scripts/compare_models.py --models mlp rnn lstm
```

### Logging

We use TensorBoard for training visualization:
```bash
tensorboard --logdir=logs/
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_environment.py -v

# Check coverage
pytest --cov=final_proj tests/
```

## 📈 Citation

If you use this code in your research, please cite:

```bibtex
@misc{docter_rl_2025,
  title={Doctor Agent for Hypotension Management using Reinforcement Learning},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/Docter_RL}
}
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This is a research project for educational purposes. **NOT FOR CLINICAL USE.** Always consult qualified medical professionals for medical decisions.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- PyGame Learning Environment (PLE) framework
- OpenAI Gym for environment standardization
- Research community for RL algorithms
