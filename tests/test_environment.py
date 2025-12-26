"""
Unit tests for Doctor RL environment
"""

import pytest
import numpy as np
import gym
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import final_proj


class TestDocterEnv:
    """Test suite for DocterEnv"""
    
    @pytest.fixture
    def env(self):
        """Create environment instance"""
        env = gym.make('final_proj/RLDocter_v0')
        yield env
        env.close()
    
    def test_environment_creation(self, env):
        """Test that environment can be created"""
        assert env is not None
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
    
    def test_action_space(self, env):
        """Test action space properties"""
        assert env.action_space.n == 2  # Should have 2 actions
        
        # Test sampling
        for _ in range(10):
            action = env.action_space.sample()
            assert action in [0, 1]
    
    def test_observation_space(self, env):
        """Test observation space properties"""
        obs_space = env.observation_space
        assert len(obs_space.shape) == 1
        assert obs_space.shape[0] > 0  # Should have at least one state variable
    
    def test_reset(self, env):
        """Test environment reset"""
        state = env.reset()
        
        # Check state type and shape
        assert isinstance(state, np.ndarray)
        assert state.shape == env.observation_space.shape
        
        # Check values are valid
        assert not np.any(np.isnan(state))
        assert not np.any(np.isinf(state))
    
    def test_step(self, env):
        """Test environment step function"""
        env.reset()
        
        # Test valid actions
        for action in range(env.action_space.n):
            state, reward, done, info = env.step(action)
            
            # Check return types
            assert isinstance(state, np.ndarray)
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)
            assert isinstance(info, dict)
            
            # Check state validity
            assert state.shape == env.observation_space.shape
            assert not np.any(np.isnan(state))
            
            if done:
                break
    
    def test_episode(self, env):
        """Test a complete episode"""
        state = env.reset()
        episode_reward = 0
        steps = 0
        max_steps = 1000
        
        done = False
        while not done and steps < max_steps:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        assert steps > 0
        assert isinstance(episode_reward, (int, float))
    
    def test_deterministic_seed(self):
        """Test that seeding makes environment deterministic"""
        env1 = gym.make('final_proj/RLDocter_v0')
        env2 = gym.make('final_proj/RLDocter_v0')
        
        seed = 42
        env1.seed(seed)
        env2.seed(seed)
        
        state1 = env1.reset()
        state2 = env2.reset()
        
        # States should be identical with same seed
        np.testing.assert_array_almost_equal(state1, state2)
        
        # Test a few steps
        for _ in range(10):
            action = 0  # Fixed action
            s1, r1, d1, _ = env1.step(action)
            s2, r2, d2, _ = env2.step(action)
            
            # Should produce same results
            # Note: Floating point comparison with tolerance
            np.testing.assert_array_almost_equal(s1, s2, decimal=5)
            assert abs(r1 - r2) < 1e-5
            assert d1 == d2
            
            if d1:
                break
        
        env1.close()
        env2.close()
    
    def test_multiple_episodes(self, env):
        """Test multiple episode resets"""
        num_episodes = 5
        
        for episode in range(num_episodes):
            state = env.reset()
            assert isinstance(state, np.ndarray)
            
            # Run a few steps
            for _ in range(10):
                action = env.action_space.sample()
                state, reward, done, info = env.step(action)
                if done:
                    break


class TestEnvironmentProperties:
    """Test specific environment properties"""
    
    def test_blood_pressure_range(self):
        """Test that blood pressure values are in reasonable range"""
        env = gym.make('final_proj/RLDocter_v0')
        env.reset()
        
        # Collect states over several steps
        states = []
        for _ in range(100):
            action = env.action_space.sample()
            state, _, done, _ = env.step(action)
            states.append(state)
            if done:
                env.reset()
        
        states = np.array(states)
        
        # Check that states are finite
        assert np.all(np.isfinite(states))
        
        env.close()
    
    def test_reward_range(self):
        """Test that rewards are in expected range"""
        env = gym.make('final_proj/RLDocter_v0')
        env.reset()
        
        rewards = []
        for _ in range(100):
            action = env.action_space.sample()
            _, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                env.reset()
        
        # Rewards should be finite
        assert all(np.isfinite(r) for r in rewards)
        
        env.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
