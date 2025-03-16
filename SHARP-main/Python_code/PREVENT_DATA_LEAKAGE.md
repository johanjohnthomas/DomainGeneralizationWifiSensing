# Preventing Data Leakage in WiFi Sensing

This document explains the problem of data leakage in WiFi sensing datasets and how to use the directory-level split feature to prevent it.

## The Problem: Data Leakage

Data leakage occurs when information from outside the training dataset is used to create the model. In WiFi sensing, this often happens when:

1. Data from the same subject or environment appears in both training and testing sets
2. Different parts of the same recording session are split between training and testing

In the original implementation, data was split within each subdirectory into train/val/test subfolders. This approach can lead to data leakage if the subdirectories contain data from the same subject/environment, as the model might "memorize" characteristics specific to that subject/environment rather than learning generalizable patterns.

## The Solution: Directory-Level Splitting

To prevent data leakage, we've implemented a directory-level splitting strategy where:

- Each subdirectory represents a unique environment or subject
- Some subdirectories are used exclusively for training
- Others are used exclusively for validation
- And others exclusively for testing

This ensures a clean separation between training, validation, and testing data, leading to more reliable performance estimates and better generalization.

## How to Use Directory-Level Splitting

### Command-Line Arguments

The following new command-line arguments have been added to `CSI_network.py`:

```
--train_subdirs    Comma-separated list of subdirectories to use for training
--val_subdirs      Comma-separated list of subdirectories to use for validation
--test_subdirs     Comma-separated list of subdirectories to use for testing
--split_mode       Mode for splitting: "directory" (default) or "file"
```

### Example Usage

You can run the code with directory-level splitting using:

```bash
python CSI_network.py <dir> <subdirs> <feature_length> <sample_length> <channels> <batch_size> <num_tot> <name_base> <activities> \
  --train_subdirs=AR1a_S,AR1b_E,AR1c_C \
  --val_subdirs=AR3a_R,AR4a_R \
  --test_subdirs=AR5a_C,AR6a_E \
  --split_mode=directory
```

Or use the provided Makefile targets:

```bash
make no_unseen_domains_directory_split
```

### Split Modes

- **directory** (recommended): Uses separate subdirectories for train/val/test to prevent data leakage
- **file** (legacy): Uses train/val/test subfolders within each subdirectory (risk of data leakage)

## Benefits of Directory-Level Splitting

1. **More Reliable Evaluation**: By completely separating train/val/test environments, performance metrics better represent the model's ability to generalize to new environments.

2. **Prevents Overfitting to Specific Environments**: The model is forced to learn patterns that generalize across environments rather than memorizing environment-specific characteristics.

3. **Better Domain Generalization**: Models trained with clean splits are more likely to perform well on truly unseen environments.

## Best Practices

1. **Subject/Environment Separation**: Ensure that data from the same subject or recording environment is contained within a single subdirectory.

2. **Balanced Classes**: Try to maintain a similar distribution of activity classes across train/val/test subdirectories.

3. **Representative Environments**: Include diverse environments in your training set to help the model generalize better.

4. **Validation**: Use the validation set to tune hyperparameters and select the best model. Only use the test set for final evaluation.

## Example Directory Structure

```
doppler_traces/
├── AR1a_S/                 # Training environment 1
│   └── train_antennas_*/   # All files here are used for training
├── AR1b_E/                 # Training environment 2
│   └── train_antennas_*/   # All files here are used for training
├── AR5a_C/                 # Validation environment
│   └── train_antennas_*/   # All files here are used for validation
└── AR7a_J1/                # Test environment
    └── train_antennas_*/   # All files here are used for testing
```

Note that with directory-level splitting, we only use the `train_antennas_*` subfolder from each directory, regardless of whether it's used for training, validation, or testing. 