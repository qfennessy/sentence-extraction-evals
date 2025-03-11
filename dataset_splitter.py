#!/usr/bin/env python3
"""
Dataset Splitter for Sentence Extraction Evaluation

This module splits datasets into train, validation, and test sets to
support prompt refinement without overfitting.
"""

import os
import json
import random
import argparse
from typing import Dict, List, Tuple
from pathlib import Path


class DatasetSplitter:
    """Splits datasets into train, validation, and test sets."""

    def __init__(
        self,
        dataset_path: str,
        output_dir: str = "dataset_splits",
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        """Initialize the dataset splitter.
        
        Args:
            dataset_path: Path to the dataset file
            output_dir: Directory to store split datasets
            train_ratio: Ratio of data for training (default: 0.7)
            validation_ratio: Ratio of data for validation (default: 0.15)
            test_ratio: Ratio of data for testing (default: 0.15)
            random_seed: Random seed for reproducibility
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Ensure ratios sum to 1.0
        total_ratio = train_ratio + validation_ratio + test_ratio
        if total_ratio != 1.0:
            # Normalize ratios
            self.train_ratio = train_ratio / total_ratio
            self.validation_ratio = validation_ratio / total_ratio
            self.test_ratio = test_ratio / total_ratio
            
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_dataset(self) -> Dict:
        """Load the dataset from the input path."""
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
    
    def split_dataset(self) -> Tuple[Dict, Dict, Dict]:
        """Split the dataset into train, validation, and test sets.
        
        Returns:
            Tuple containing train, validation, and test dataset dictionaries
        """
        # Load the dataset
        data = self.load_dataset()
        
        if "sentences" not in data:
            raise ValueError(f"Dataset at {self.dataset_path} does not contain 'sentences' key")
        
        sentences = data["sentences"]
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        
        # Shuffle sentences
        shuffled_sentences = sentences.copy()
        random.shuffle(shuffled_sentences)
        
        # Calculate split indices
        total_sentences = len(shuffled_sentences)
        train_size = int(total_sentences * self.train_ratio)
        validation_size = int(total_sentences * self.validation_ratio)
        
        # Split the sentences
        train_sentences = shuffled_sentences[:train_size]
        validation_sentences = shuffled_sentences[train_size:train_size + validation_size]
        test_sentences = shuffled_sentences[train_size + validation_size:]
        
        # Create output datasets
        train_data = {"sentences": train_sentences}
        validation_data = {"sentences": validation_sentences}
        test_data = {"sentences": test_sentences}
        
        return train_data, validation_data, test_data
    
    def save_splits(self) -> Tuple[str, str, str]:
        """Split and save datasets to output directory.
        
        Returns:
            Tuple of paths to the saved train, validation, and test datasets
        """
        # Split the dataset
        train_data, validation_data, test_data = self.split_dataset()
        
        # Create dataset-specific directory
        dataset_name = self.dataset_path.stem
        dataset_dir = self.output_dir / dataset_name
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save the splits
        train_path = dataset_dir / f"{dataset_name}_train.json"
        validation_path = dataset_dir / f"{dataset_name}_validation.json"
        test_path = dataset_dir / f"{dataset_name}_test.json"
        
        with open(train_path, 'w') as f:
            json.dump(train_data, f, indent=2)
            
        with open(validation_path, 'w') as f:
            json.dump(validation_data, f, indent=2)
            
        with open(test_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Create metadata file
        metadata = {
            "original_dataset": str(self.dataset_path),
            "dataset_name": dataset_name,
            "train_size": len(train_data["sentences"]),
            "validation_size": len(validation_data["sentences"]),
            "test_size": len(test_data["sentences"]),
            "train_ratio": self.train_ratio,
            "validation_ratio": self.validation_ratio,
            "test_ratio": self.test_ratio,
            "random_seed": self.random_seed,
            "train_path": str(train_path),
            "validation_path": str(validation_path),
            "test_path": str(test_path)
        }
        
        metadata_path = dataset_dir / "split_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset {dataset_name} split into:")
        print(f"  - Train: {len(train_data['sentences'])} sentences")
        print(f"  - Validation: {len(validation_data['sentences'])} sentences")
        print(f"  - Test: {len(test_data['sentences'])} sentences")
        print(f"Files saved to {dataset_dir}")
        
        return str(train_path), str(validation_path), str(test_path)


def main():
    """Run the dataset splitter from command line."""
    parser = argparse.ArgumentParser(description="Split datasets for prompt refinement")
    parser.add_argument("datasets", nargs="+", help="Dataset files to split")
    parser.add_argument("--output-dir", default="dataset_splits", help="Directory to store split datasets")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Ratio of data for training")
    parser.add_argument("--validation-ratio", type=float, default=0.15, help="Ratio of data for validation")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Ratio of data for testing")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    for dataset_path in args.datasets:
        splitter = DatasetSplitter(
            dataset_path=dataset_path,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed
        )
        
        splitter.save_splits()


if __name__ == "__main__":
    main()