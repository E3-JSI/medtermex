# Makefile for project

.PHONY: setup clean lint help

# Default target
.DEFAULT_GOAL := help

# Help command
help:
	@echo "Available commands:"
	@echo "  setup         : Set up the development environment"
	@echo "  clean         : Clean up generated files and caches"
	@echo "  lint          : Run linters (black, isort, flake8)"
	@echo "  help          : Show this help message"

# Environment setup
setup:
	@echo "Setting up the development environment..."
	bash scripts/setup_env.sh

# Clean up
clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ .pytest_cache .coverage
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Linting
lint:
	@echo "Running linters..."
	black src
	isort src
	flake8 src
