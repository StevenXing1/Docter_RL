# Contributing to Doctor RL

Thank you for considering contributing to this project! 🎉

## How to Contribute

### Reporting Issues

- Use the issue tracker to report bugs or suggest features
- Provide clear descriptions and steps to reproduce bugs
- Include system information (OS, Python version, PyTorch version)

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, readable code
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests** before submitting
   ```bash
   pytest tests/ -v
   ```

4. **Format your code**
   ```bash
   black final_proj/ scripts/ tests/
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add: brief description of your changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** with a clear title and description

## Code Style

- Follow PEP 8 style guidelines
- Use `black` for code formatting
- Use type hints where appropriate
- Write descriptive docstrings

## Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting
- Aim for high test coverage

## Documentation

- Update README.md if needed
- Add docstrings to new functions/classes
- Update configuration examples if adding new features

## Questions?

Feel free to open an issue for questions or discussions!
