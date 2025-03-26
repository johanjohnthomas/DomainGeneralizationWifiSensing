# Domain Generalization for WiFi Sensing

This repository contains an implementation of various approaches for domain generalization in WiFi sensing, with a primary focus on the SHARP (Spatial-Harmonized Activity Recognition Pipeline) framework. The project enables robust human activity recognition across different physical environments using WiFi Channel State Information (CSI).

## Overview

SHARP is designed to address the challenge of domain shift in WiFi sensing systems, where models trained in one environment often perform poorly when deployed in new, unseen environments. The framework implements several techniques to mitigate this issue:

- Phase sanitization and signal preprocessing to remove hardware-induced artifacts
- Doppler feature extraction that captures motion-related information
- Multiple deep learning models for cross-domain activity recognition

We find it interesting to evaluate the effectiveness of these models on different experiments i.e. Leave-one-out, source scaling etc.

## Available Models

The pipeline supports three different model architectures:

1. **LSTM-CNN** - A hybrid model combining Convolutional Neural Networks for spatial feature extraction with Long Short-Term Memory networks for temporal analysis
2. **SHARP** - The Spatial-Harmonized Activity Recognition Pipeline using an Inception-ResNet style architecture optimized for domain generalization
3. **WiDAR** - A GRU-CNN hybrid model based on WiFi Doppler activity recognition techniques

## Pipeline Structure

The project implements a comprehensive pipeline controlled through a Makefile that handles:

1. **Data Processing**
   - Phase sanitization of raw CSI data
   - H-matrix estimation
   - Signal reconstruction
   - Doppler computation

2. **Dataset Creation**
   - Activity filtering
   - Training and test dataset generation
   - Data balancing options

3. **Model Training and Evaluation**
   - Training with multiple domain configurations
   - Testing on unseen domains
   - Metrics calculation and visualization

## Research Questions

The Makefile organizes experiments into different research questions (RQ):

- **RQ1**: Generalization gap and leave-one-out domain evaluation
- **RQ4**: Source domain scaling with varying amounts of training data
- **RQ5**: Component analysis (antenna randomization, phase sanitization)
- **RQ6**: Target domain scaling with varying amounts of adaptation data
- **RQ7**: Temporal analysis (same-day vs. different-day evaluations)

## Utility and Visualization Scripts

The project includes several utility scripts for data processing and visualization:

- **plots_utility.py**: Core plotting functions for signal visualization
- **generate_all_plots.py**: Comprehensive visualization generator
- **CSI_doppler_plots_antennas.py**: Antenna-specific Doppler visualizations
- **CSI_network_metrics.py**: Performance metrics calculation
- **CSI_network_metrics_plot.py**: Performance visualization tools

These scripts generate various plots including:
- Confusion matrices
- Doppler activity visualizations
- Phase and amplitude comparisons
- Domain-specific performance analysis

## Usage

### Basic Pipeline Execution

To run the complete pipeline with default settings:

```bash
cd SHARP-main/Python_code
make create_structure
make run_complete MODEL_NAME="custom_model" TRAIN_DOMAINS="AR1a,AR5a" TEST_DOMAINS="AR6a" MODEL_TYPE="inc_res"
```

### Model Selection

To select a specific model architecture:

```bash
# For LSTM-CNN model
make run_complete MODEL_TYPE="lstm_cnn" ...

# For SHARP model (Inception-ResNet style)
make run_complete MODEL_TYPE="inc_res" ...

# For WiDAR model (GRU-CNN)
make run_complete MODEL_TYPE="gru_cnn" ...
```

### Experiment Categories

```bash
# Leave-one-out domain evaluation
make leave_one_out

# Domain scaling experiments
make varying_sources

# Target domain scaling
make target_scaling_all

# Temporal analysis experiments
make temporal_analysis_all
```

## Requirements

- Python 3.6+
- TensorFlow 2.x
- NumPy, Matplotlib, SciPy
- CUDA-enabled GPU recommended for model training

## Dataset Structure

The input data should be organized in a specific directory structure:
- Each environment is designated with an identifier (e.g., AR1a, AR5a)
- Activities are labeled with specific codes (E: empty, J: jumping, L: walking in loop, R: running, W: walking)

## Results and Analysis

Results are systematically organized in the `results` directory, categorized by research question and experiment type. Metrics and visualizations are automatically generated after each experiment.

The results directory structure is organized as follows:

```
results/
├── RQ1_generalization/
│   ├── generalization_gap/
│   │   ├── MODEL_NAME/
│   │   │   ├── metrics/
│   │   │   │   ├── confusion_matrix_*.txt
│   │   │   │   └── performance_metrics_*.txt
│   │   │   └── plots/
│   │   │       ├── confusion_matrix_*.png
│   │   │       └── activity_*.png
│   └── leave_one_out/
│       ├── no_bedroom/
│       ├── no_kitchen/
│       ├── no_lab/
│       ├── no_living/
│       ├── no_office/
│       └── no_semi/
├── RQ4_source_scaling/
│   ├── source_1/
│   ├── source_2/
│   ├── source_3/
│   └── source_4/
├── RQ5_component_analysis/
│   ├── no_antenna_randomization/
│   └── no_phase_sanitization/
├── RQ6_target_scaling/
│   ├── target_1/
│   ├── target_2/
│   ├── target_3/
│   ├── target_4/
│   ├── target_5/
│   └── target_6/
└── RQ7_temporal_analysis/
    ├── same_day/
    │   ├── 1/
    │   └── 2/
    └── different_day/
        ├── 1/
        └── 2/
```

Each experiment directory contains metrics and plots that analyze the model performance for that specific configuration.

## Citations

If you use this code or find it helpful in your research, please cite the following papers:

For the SHARP framework:
```
@article{meneghello2022sharp,
  author={Meneghello, Francesca and Garlisi, Domenico and Dal Fabbro, Nicol\o' and Tinnirello, Ilenia and Rossi, Michele},
  journal={IEEE Transactions on Mobile Computing}, 
  title={{SHARP: Environment and Person Independent Activity Recognition with Commodity IEEE 802.11 Access Points}}, 
  year={2023},
  volume={22},
  number={10},
  pages={6160-6175}
  }
```

For the WiSR approach:
```
@ARTICLE{10172243,
  author={Liu, Shijia and Chen, Zhenghua and Wu, Min and Liu, Chang and Chen, Liangyin},
  journal={IEEE Transactions on Mobile Computing}, 
  title={WiSR: Wireless Domain Generalization Based on Style Randomization}, 
  year={2023},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TMC.2023.3292229}}
```
