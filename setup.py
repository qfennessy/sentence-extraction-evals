#!/usr/bin/env python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sentence-extraction",
    version="0.1.0",
    author="Anthropic",
    author_email="info@anthropic.com",
    description="A package for evaluating language models on sentence extraction tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anthropic/sentence-extraction-evals",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "anthropic>=0.3.0",
        "openai>=1.0.0",
        "click>=8.0.0",
        "pandas>=1.0.0",
        "tqdm>=4.60.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pre-commit>=2.20.0",
            "black>=23.1.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "flake8-docstrings>=1.7.0",
            "flake8-bugbear>=23.1.20",
            "mypy>=1.0.1",
            "pytest>=7.2.1",
            "pytest-cov>=4.0.0",
            "bandit>=1.7.5",
            "ruff>=0.0.54",
            "nbqa>=1.7.0",
            "types-requests>=2.28.11.17",
            "types-PyYAML>=6.0.12.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "extraction-eval=sentence_extraction.scripts.extraction_eval:main",
        ],
    },
)