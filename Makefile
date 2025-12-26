# Makefile for Doctor RL project

.PHONY: help install install-dev test lint format clean train evaluate

help:
	@echo "Available commands:"
	@echo "  make install      - Install package and dependencies"
	@echo "  make install-dev  - Install package with development dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linting checks"
	@echo "  make format       - Format code with black"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make train        - Train MLP model"
	@echo "  make evaluate     - Evaluate trained model"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=final_proj --cov-report=html --cov-report=term

lint:
	flake8 final_proj/ scripts/ tests/
	pylint final_proj/ scripts/

format:
	black final_proj/ scripts/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

train:
	python scripts/train.py --config configs/mlp_config.yaml

evaluate:
	python scripts/evaluate.py --model-path models/best_model.pth --model-type mlp
