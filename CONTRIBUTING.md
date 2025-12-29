# Contributing to WHO Handwashing Monitoring System

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, GPU/CPU)
- Error messages or logs

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:
- Clear description of the enhancement
- Use case / motivation
- Potential implementation approach
- Any relevant examples or references

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/handwash-monitoring.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation if needed

4. **Test your changes**
   ```bash
   python preprocess_data.py  # If data pipeline changed
   python train_ensemble.py   # If model training changed
   python evaluate_ensemble.py # If evaluation changed
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "Add: Brief description of feature"
   ```

6. **Push and create Pull Request**
   ```bash
   git push origin feature/YourFeatureName
   ```

## 📝 Code Style

- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and modular

### Example:
```python
def normalize_landmarks(landmarks):
    """
    Normalize hand landmarks to be position/scale invariant.
    
    Args:
        landmarks (np.ndarray): Shape (21, 3) raw landmark coordinates
        
    Returns:
        np.ndarray: Shape (63,) normalized and flattened landmarks
    """
    # Implementation...
```

## 🧪 Testing

Before submitting:
- [ ] Code runs without errors
- [ ] Results are reproducible
- [ ] Documentation is updated
- [ ] No unnecessary files added

## 📋 Areas for Contribution

**High Priority:**
- [ ] Real-time inference optimization
- [ ] Mobile deployment (TFLite conversion)
- [ ] Data augmentation strategies
- [ ] Attention visualization tools

**Medium Priority:**
- [ ] Multi-camera fusion
- [ ] Extended temporal context experiments
- [ ] Hyperparameter optimization
- [ ] Alternative architectures (Transformers)

**Documentation:**
- [ ] Tutorial notebooks
- [ ] Video demos
- [ ] API documentation
- [ ] Deployment guides

## ❓ Questions?

Feel free to open an issue for any questions or clarifications!

Thank you for contributing! 🎉