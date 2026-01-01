#!/usr/bin/env python3
"""
Create a small test dataset with train and test splits from the full OCT dataset.
This is for testing purposes and can be uploaded to git repo.
"""

import argparse
from pathlib import Path
from datasets import load_from_disk, DatasetDict

def create_test_dataset(
    train_source_path: str,
    test_source_path: str,
    output_path: str,
    train_samples: int = 100,
    test_samples: int = 50,
    seed: int = 42
):
    """
    Create a test dataset with train and test splits by sampling from source datasets.
    Creates a DatasetDict with 'train' and 'test' splits.
    
    Args:
        train_source_path: Path to the source train dataset
        test_source_path: Path to the source test dataset
        output_path: Path to save the test dataset
        train_samples: Number of samples for train split
        test_samples: Number of samples for test split
        seed: Random seed for reproducibility
    """
    print(f"Loading train dataset from: {train_source_path}")
    train_dataset = load_from_disk(train_source_path)
    
    print(f"Loading test dataset from: {test_source_path}")
    test_dataset = load_from_disk(test_source_path)
    
    # Handle DatasetDict - extract the appropriate split
    if isinstance(train_dataset, DatasetDict):
        if 'train' in train_dataset:
            train_data = train_dataset['train']
        else:
            # Get the first split
            split_name = list(train_dataset.keys())[0]
            print(f"Using split '{split_name}' from train dataset")
            train_data = train_dataset[split_name]
    else:
        train_data = train_dataset
    
    if isinstance(test_dataset, DatasetDict):
        if 'test' in test_dataset:
            test_data = test_dataset['test']
        else:
            # Get the first split
            split_name = list(test_dataset.keys())[0]
            print(f"Using split '{split_name}' from test dataset")
            test_data = test_dataset[split_name]
    else:
        test_data = test_dataset
    
    # Check available samples
    train_total = len(train_data)
    test_total = len(test_data)
    
    print(f"\n=== Source Dataset Info ===")
    print(f"Train split: {train_total} samples available")
    print(f"Test split: {test_total} samples available")
    
    # Sample train split
    if train_samples > train_total:
        print(f"Warning: Requested {train_samples} train samples but only {train_total} available.")
        print(f"Using all {train_total} samples.")
        train_samples = train_total
    
    print(f"\nSampling {train_samples} samples from train split with seed={seed}...")
    sampled_train = train_data.shuffle(seed=seed).select(range(train_samples))
    
    # Sample test split
    if test_samples > test_total:
        print(f"Warning: Requested {test_samples} test samples but only {test_total} available.")
        print(f"Using all {test_total} samples.")
        test_samples = test_total
    
    print(f"Sampling {test_samples} samples from test split with seed={seed}...")
    sampled_test = test_data.shuffle(seed=seed).select(range(test_samples))
    
    # Create DatasetDict with train and test splits
    test_dataset_dict = DatasetDict({
        'train': sampled_train,
        'test': sampled_test
    })
    
    # Save the test dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving test dataset to: {output_path}")
    test_dataset_dict.save_to_disk(str(output_path))
    
    print(f"\n✓ Test dataset created successfully!")
    print(f"  - Split 'train': {len(sampled_train)} samples")
    print(f"  - Split 'test': {len(sampled_test)} samples")
    print(f"  - Output path: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a small test dataset with train and test splits")
    parser.add_argument(
        "--train_source",
        type=str,
        default="/data/datasets/OmniMedVQA/OmniMedVQA/open_access_sft_data_hf_modality_OCT_(Optical_Coherence_Tomography_train",
        help="Path to source train dataset"
    )
    parser.add_argument(
        "--test_source",
        type=str,
        default="/data/datasets/OmniMedVQA/OmniMedVQA/open_access_sft_data_hf_modality_OCT_(Optical_Coherence_Tomography_test",
        help="Path to source test dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/fayang/Med-R1/data/open_access_sft_data_hf_modality_OCT_train_test",
        help="Path to save test dataset"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=100,
        help="Number of samples for train split"
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=50,
        help="Number of samples for test split"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    
    args = parser.parse_args()
    create_test_dataset(
        train_source_path=args.train_source,
        test_source_path=args.test_source,
        output_path=args.output,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        seed=args.seed
    )

