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
    entry_points={
        "console_scripts": [
            "extraction-eval=sentence_extraction.scripts.extraction_eval:main",
        ],
    },
)