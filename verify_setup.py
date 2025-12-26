#!/usr/bin/env python
"""
Quick demo script to verify installation and run basic tests
"""

import sys
import subprocess
from pathlib import Path


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def check_python_version():
    """Check Python version"""
    print_section("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠ Warning: Python 3.8+ recommended")
        return False
    else:
        print("✓ Python version OK")
        return True


def check_package_installed():
    """Check if package is installed"""
    print_section("Package Installation")
    try:
        import final_proj
        print(f"✓ final_proj package found (version {final_proj.__version__})")
        return True
    except ImportError:
        print("✗ final_proj not installed")
        print("Run: pip install -e .")
        return False


def check_dependencies():
    """Check key dependencies"""
    print_section("Dependencies")
    
    deps = {
        'gym': 'OpenAI Gym',
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pygame': 'Pygame',
        'yaml': 'PyYAML',
    }
    
    all_ok = True
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} not found")
            all_ok = False
    
    return all_ok


def test_environment():
    """Test environment creation"""
    print_section("Environment Test")
    
    try:
        import gym
        import final_proj
        
        env = gym.make('final_proj/RLDocter_v0')
        print(f"✓ Environment created")
        print(f"  State space: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space.n} actions")
        
        # Test reset
        state = env.reset()
        print(f"✓ Reset successful, state shape: {state.shape}")
        
        # Test step
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print(f"✓ Step successful, reward: {reward:.2f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False


def test_model_import():
    """Test model imports"""
    print_section("Model Import Test")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.train import PolicyMLP, PolicyRNN, PolicyLSTM
        
        print("✓ PolicyMLP imported")
        print("✓ PolicyRNN imported")
        print("✓ PolicyLSTM imported")
        
        # Try creating a model
        model = PolicyMLP(8, 2, 64)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ MLP model created ({param_count:,} parameters)")
        
        return True
        
    except Exception as e:
        print(f"✗ Model import failed: {e}")
        return False


def run_quick_training():
    """Run quick training test"""
    print_section("Quick Training Test (10 episodes)")
    
    try:
        print("Training MLP for 10 episodes...")
        result = subprocess.run(
            ['python', 'scripts/train.py', '--model', 'mlp', '--episodes', '10'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✓ Training completed successfully")
            return True
        else:
            print(f"✗ Training failed")
            if result.stderr:
                print(result.stderr[-500:])  # Last 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Training timed out")
        return False
    except Exception as e:
        print(f"✗ Training test failed: {e}")
        return False


def main():
    """Run all checks"""
    print("\n" + "🏥 Doctor RL - Installation & Setup Verification")
    
    results = {
        'Python Version': check_python_version(),
        'Package Installation': check_package_installed(),
        'Dependencies': check_dependencies(),
        'Environment': test_environment(),
        'Models': test_model_import(),
    }
    
    # Summary
    print_section("SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for check, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {check}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! Setup is complete.")
        print("\nNext steps:")
        print("  1. Train a model: python scripts/train.py --model mlp --episodes 1000")
        print("  2. Evaluate: python scripts/evaluate.py --model-path models/mlp/best_model.pth --model-type mlp")
        print("  3. Visualize: python scripts/visualize.py --mode training --input models/mlp/final_model.pth")
    else:
        print("\n⚠ Some checks failed. Please install missing dependencies:")
        print("  pip install -e .")
        print("  pip install -r requirements.txt")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
