# Nude Generator Project

A production-ready AI-powered nude image generator using Stable Diffusion inpainting techniques.

## 🚀 Features

- **High-Quality Generation**: Uses state-of-the-art Stable Diffusion models for realistic results
- **Intelligent Masking**: Automatic clothing detection and smart mask generation
- **Pose Preservation**: Maintains body pose and structure during generation
- **Face & Hand Protection**: Preserves facial features and hand details
- **Batch Processing**: Process multiple images efficiently
- **Multiple Quality Settings**: Fast, default, and high-quality generation modes
- **Customizable Prompts**: Advanced prompting system for fine-tuned control
- **Production Ready**: Comprehensive error handling, logging, and configuration

## 📁 Project Structure

```
nude_generator_project/
├── src/nude_generator/           # Main source code
│   ├── __init__.py
│   ├── core/                     # Core generation logic
│   │   ├── __init__.py
│   │   ├── generator.py          # Main generator class
│   │   ├── advanced_generator.py # Advanced features
│   │   └── pipeline.py           # Pipeline management
│   ├── models/                   # Model management
│   │   ├── __init__.py
│   │   ├── model_loader.py       # Model loading utilities
│   │   └── model_config.py       # Model configurations
│   ├── processing/               # Image processing
│   │   ├── __init__.py
│   │   ├── preprocessor.py       # Image preprocessing
│   │   ├── mask_generator.py     # Mask generation
│   │   └── postprocessor.py      # Post-processing
│   ├── detection/                # Detection modules
│   │   ├── __init__.py
│   │   ├── clothing_detector.py  # Clothing detection
│   │   ├── pose_detector.py      # Pose detection
│   │   └── face_detector.py      # Face detection
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── config.py             # Configuration management
│       ├── logger.py             # Logging utilities
│       └── helpers.py            # Helper functions
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_generator.py         # Generator tests
│   ├── test_processing.py        # Processing tests
│   └── test_integration.py       # Integration tests
├── examples/                     # Usage examples
│   ├── basic_usage.py            # Basic examples
│   ├── advanced_usage.py         # Advanced examples
│   └── batch_processing.py       # Batch processing examples
├── scripts/                      # Utility scripts
│   ├── setup.py                  # Setup script
│   ├── download_models.py        # Model download script
│   └── benchmark.py              # Performance benchmarking
├── configs/                      # Configuration files
│   ├── default.yaml              # Default configuration
│   ├── high_quality.yaml         # High quality settings
│   └── fast.yaml                 # Fast generation settings
├── docs/                         # Documentation
│   ├── installation.md           # Installation guide
│   ├── usage.md                  # Usage documentation
│   ├── api.md                    # API reference
│   └── troubleshooting.md        # Troubleshooting guide
├── data/                         # Data directory
│   ├── input/                    # Input images
│   ├── output/                   # Generated outputs
│   └── models/                   # Downloaded models
├── assets/                       # Project assets
│   └── sample_images/            # Sample test images
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── pyproject.toml               # Modern Python packaging
└── README.md                     # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended, 8GB+ VRAM)
- 16GB+ RAM

### Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd nude_generator_project

# Install dependencies
pip install -r requirements.txt

# Download models (optional, will auto-download on first use)
python scripts/download_models.py
```

### Development Install

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

## 🚀 Quick Start

### Basic Usage

```python
from nude_generator import NudeGenerator

# Initialize generator
generator = NudeGenerator()

# Generate nude image
result = generator.generate_nude(
    image="path/to/image.jpg",
    quality="default"
)

# Save result
result.save("output.png")
```

### Advanced Usage

```python
from nude_generator import AdvancedNudeGenerator

# Initialize with advanced features
generator = AdvancedNudeGenerator(
    enable_pose_preservation=True,
    enable_face_protection=True
)

# Generate with custom settings
result = generator.generate_nude_advanced(
    image="path/to/image.jpg",
    quality="high",
    preserve_face=True,
    preserve_hands=True,
    custom_prompt="realistic nude body, natural lighting"
)
```

### Command Line Usage

```bash
# Basic generation
python -m nude_generator.cli generate input.jpg -o output.png

# Batch processing
python -m nude_generator.cli batch input_folder/ -o output_folder/

# Advanced options
python -m nude_generator.cli generate input.jpg \
    --quality high \
    --preserve-face \
    --preserve-hands \
    --prompt "custom prompt here"
```

## 📖 Documentation

- [Installation Guide](docs/installation.md)
- [Usage Documentation](docs/usage.md)
- [API Reference](docs/api.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_generator.py

# Run with coverage
python -m pytest tests/ --cov=nude_generator
```

## ⚙️ Configuration

The project uses YAML configuration files for different settings:

- `configs/default.yaml` - Default settings
- `configs/high_quality.yaml` - High quality generation
- `configs/fast.yaml` - Fast generation

## 🔧 Development

### Code Style

```bash
# Format code
black src/ tests/ examples/

# Lint code
flake8 src/ tests/ examples/

# Type checking
mypy src/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## ⚠️ Ethical Considerations

This tool is designed for legitimate artistic, educational, and research purposes. Users must:

- Ensure they have proper consent for any images processed
- Comply with local laws and regulations
- Use the tool responsibly and ethically
- Respect privacy and dignity of individuals

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Stability AI for Stable Diffusion models
- Hugging Face for the diffusers library
- The open-source AI community

## 📞 Support

- Create an issue for bug reports
- Check the [troubleshooting guide](docs/troubleshooting.md)
- Review existing issues before creating new ones

---

**Disclaimer**: This software is provided for educational and research purposes. Users are responsible for ensuring ethical and legal use of the generated content.

