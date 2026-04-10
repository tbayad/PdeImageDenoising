
# PDE Image Denoising - Perona-Malik Diffusion

A Python implementation of the Perona-Malik anisotropic diffusion algorithm for image denoising using partial differential equations.

## Overview

This project implements the Perona-Malik diffusion model, an edge-preserving image denoising technique that uses anisotropic diffusion to reduce noise while maintaining important image features.

## Features

- **Edge-preserving denoising**: Reduces noise while keeping edges sharp
- **Anisotropic diffusion**: Directional smoothing based on image gradients
- **Flexible dataset support**: Works with various image formats
- **Noise analysis**: Compute and compare noise metrics before and after denoising

## Installation

### Prerequisites
- Python 3.12+
- [List dependencies here once finalized]

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd PdeImageDenoising

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
# Example usage will go here
```

### Demo

Run the demo script to see the algorithm in action:

```bash
python demo.py
```

## How It Works

The Perona-Malik algorithm processes images through the following steps:

1. **Load dataset** - Input image data from files or dataset sources
2. **Convert to grayscale** - Transform RGB tensor to grayscale matrix
3. **Add noise** - Apply noise to test algorithm robustness
4. **Compute noise ratios** - Calculate baseline noise metrics
5. **Apply diffusion algorithm** - Run Perona-Malik denoising process
6. **Compare results** - Evaluate noise reduction effectiveness

## Project Structure

```
PdeImageDenoising/
├── src/
│   └── main.py           # Core implementation
├── scripts/              # Utility scripts
├── demo.py              # Demo/example usage
├── README.md            # This file
└── pyproject.toml       # Project configuration
```

## Results

[Add sample images or metrics comparing noisy vs. denoised outputs]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify license - e.g., MIT, Apache 2.0, etc.]
