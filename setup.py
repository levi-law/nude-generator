#!/usr/bin/env python3
"""
Setup script for the Nude Generator package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="nude-generator",
    version="1.0.0",
    author="AI Research Team",
    author_email="research@example.com",
    description="AI-powered nude image generator using Stable Diffusion inpainting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/nude-generator",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "advanced": [
            "mediapipe>=0.10.0",
            "controlnet-aux>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nude-generator=nude_generator.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "nude_generator": [
            "configs/*.yaml",
            "assets/*",
        ],
    },
    keywords="ai, machine learning, image generation, stable diffusion, inpainting, nude",
    project_urls={
        "Bug Reports": "https://github.com/username/nude-generator/issues",
        "Source": "https://github.com/username/nude-generator",
        "Documentation": "https://github.com/username/nude-generator/docs",
    },
)

