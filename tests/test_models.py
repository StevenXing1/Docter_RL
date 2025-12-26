"""
Unit tests for RL models
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import models from training script
from scripts.train import PolicyMLP, PolicyRNN, PolicyLSTM


class TestPolicyModels:
    """Test suite for policy neural networks"""

    @pytest.fixture(params=[8, 16])
    def state_size(self, request):
        return request.param

    @pytest.fixture(params=[2, 4])
    def action_size(self, request):
        return request.param

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_mlp_creation(self, state_size, action_size):
        """Test MLP policy creation"""
        model = PolicyMLP(state_size, action_size, h_size=64)

        assert isinstance(model, nn.Module)
        assert model.fc1.in_features == state_size
        assert model.fc3.out_features == action_size

    def test_rnn_creation(self, state_size, action_size):
        """Test RNN policy creation"""
        model = PolicyRNN(state_size, action_size, h_size=32, num_layers=2)

        assert isinstance(model, nn.Module)
        assert model.rnn.input_size == state_size
        assert model.fc.out_features == action_size

    def test_lstm_creation(self, state_size, action_size):
        """Test LSTM policy creation"""
        model = PolicyLSTM(state_size, action_size, h_size=32, num_layers=2)

        assert isinstance(model, nn.Module)
        assert model.lstm.input_size == state_size
        assert model.fc.out_features == action_size

    def test_mlp_forward(self, state_size, action_size, device):
        """Test MLP forward pass"""
        model = PolicyMLP(state_size, action_size)
        model.to(device)

        # Create dummy input
        state = torch.randn(1, state_size).to(device)

        # Forward pass
        output = model(state)

        # Check output shape and properties
        assert output.shape == (1, action_size)
        assert torch.allclose(
            output.sum(), torch.tensor(1.0), atol=1e-5
        )  # Softmax sum to 1
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Valid probabilities

    def test_rnn_forward(self, state_size, action_size, device):
        """Test RNN forward pass"""
        model = PolicyRNN(state_size, action_size, h_size=32)
        model.to(device)

        state = torch.randn(1, state_size).to(device)
        output, hidden = model(state)

        assert output.shape == (1, action_size)
        assert torch.allclose(output.sum(), torch.tensor(1.0), atol=1e-5)
        assert hidden is not None

    def test_lstm_forward(self, state_size, action_size, device):
        """Test LSTM forward pass"""
        model = PolicyLSTM(state_size, action_size, h_size=32)
        model.to(device)

        state = torch.randn(1, state_size).to(device)
        output, hidden = model(state)

        assert output.shape == (1, action_size)
        assert torch.allclose(output.sum(), torch.tensor(1.0), atol=1e-5)
        assert hidden is not None
        assert len(hidden) == 2  # LSTM returns (h, c)

    def test_mlp_act(self, state_size, action_size, device):
        """Test MLP action selection"""
        model = PolicyMLP(state_size, action_size)
        model.to(device)

        state = np.random.randn(state_size).astype(np.float32)

        action, log_prob = model.act(state, device)

        assert isinstance(action, int)
        assert 0 <= action < action_size
        assert isinstance(log_prob, torch.Tensor)

    def test_rnn_act(self, state_size, action_size, device):
        """Test RNN action selection"""
        model = PolicyRNN(state_size, action_size)
        model.to(device)

        state = np.random.randn(state_size).astype(np.float32)

        action, log_prob, hidden = model.act(state, device)

        assert isinstance(action, int)
        assert 0 <= action < action_size
        assert isinstance(log_prob, torch.Tensor)
        assert hidden is not None

    def test_lstm_act(self, state_size, action_size, device):
        """Test LSTM action selection"""
        model = PolicyLSTM(state_size, action_size)
        model.to(device)

        state = np.random.randn(state_size).astype(np.float32)

        action, log_prob, hidden = model.act(state, device)

        assert isinstance(action, int)
        assert 0 <= action < action_size
        assert isinstance(log_prob, torch.Tensor)
        assert hidden is not None

    def test_model_training(self, state_size, action_size, device):
        """Test that models can be trained"""
        model = PolicyMLP(state_size, action_size)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Dummy training step
        state = torch.randn(1, state_size).to(device)
        target = torch.zeros(1, action_size).to(device)
        target[0, 0] = 1.0  # One-hot target

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Training step
        optimizer.zero_grad()
        output = model(state)
        loss = -torch.sum(target * torch.log(output + 1e-8))
        loss.backward()
        optimizer.step()

        # Check that parameters changed
        changed = False
        for initial, current in zip(initial_params, model.parameters()):
            if not torch.allclose(initial, current):
                changed = True
                break

        assert changed, "Model parameters should change after training"

    def test_model_save_load(self, state_size, action_size, device, tmp_path):
        """Test model saving and loading"""
        model = PolicyMLP(state_size, action_size)
        model.to(device)

        # Save model
        save_path = tmp_path / "test_model.pth"
        torch.save(model.state_dict(), save_path)

        # Create new model and load weights
        new_model = PolicyMLP(state_size, action_size)
        new_model.load_state_dict(torch.load(save_path))
        new_model.to(device)

        # Test that outputs are identical
        state = torch.randn(1, state_size).to(device)
        output1 = model(state)
        output2 = new_model(state)

        assert torch.allclose(output1, output2)


class TestModelArchitectures:
    """Test specific architectural properties"""

    def test_mlp_parameter_count(self):
        """Test MLP has reasonable number of parameters"""
        model = PolicyMLP(8, 2, h_size=64)
        param_count = sum(p.numel() for p in model.parameters())

        # Should have reasonable number of parameters
        assert param_count > 100
        assert param_count < 100000

    def test_rnn_hidden_state(self):
        """Test RNN maintains hidden state correctly"""
        model = PolicyRNN(8, 2, h_size=32, num_layers=2)

        state = torch.randn(1, 8)
        _, hidden1 = model(state, None)
        _, hidden2 = model(state, hidden1)

        # Hidden states should be different
        assert not torch.allclose(hidden1, hidden2)

    def test_lstm_memory(self):
        """Test LSTM maintains cell state"""
        model = PolicyLSTM(8, 2, h_size=32, num_layers=2)

        state = torch.randn(1, 8)
        _, (h1, c1) = model(state, None)
        _, (h2, c2) = model(state, (h1, c1))

        # Cell states should be different
        assert not torch.allclose(c1, c2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
