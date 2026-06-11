"""SDL Scheduling Project Setup."""

from setuptools import setup, find_packages
from pathlib import Path

# README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

setup(
    name="sdl_scheduling",
    version="0.1.0",
    description="High-Throughput Self-Driving Lab Dynamic Scheduling with Hierarchical RL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    python_requires=">=3.9",

    # 
    packages=find_packages(exclude=['tests', 'tests.*', 'data', 'checkpoints']),

    # 
    install_requires=[
        "torch==2.0.1",
        "torch-geometric==2.3.1",
        "numpy==1.24.3",
        "pandas==2.0.2",
        "matplotlib==3.7.1",
        "scipy==1.10.1",
        "scikit-learn==1.2.2",
        "tensorboard==2.13.0",
        "pyyaml==6.0",
        "tqdm==4.65.0",
        "gymnasium==0.29.1",  # gym
        "seaborn==0.12.2",
        "pytest==7.3.1",
    ],

    # 
    extras_require={
        "milp": ["gurobipy==10.0.1"],
        "dev": [
            "pytest-cov==4.1.0",
            "black==23.3.0",
            "flake8==6.0.0",
        ],
    },

    # 
    entry_points={
        'console_scripts': [
            'sdl-train=training.train_lower:main',
            'sdl-eval=evaluation.evaluate:main',
        ],
    },
)