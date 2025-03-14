# Model Architecture Update - 4-Channel Multi-Antenna Input

## Critical Changes Made

The CSI network architecture has been updated to process all 4 antennas as channels simultaneously, which significantly improves classification accuracy by leveraging cross-antenna correlations.

### Key Changes:

1. Changed model input shape from `(340, 100, 1)` to `(340, 100, 4)`
2. Modified the data loading pipeline to process all antennas together
3. Removed the workaround that was discarding 3 out of 4 antennas

## How to Retrain Your Model

Before running the test script, you must retrain your model with the new architecture:

```bash
# Navigate to the Python_code directory
cd /path/to/SHARP-main/Python_code

# Run the training script with your usual parameters
python CSI_network.py <dir> <subdirs> <feature_length> <sample_length> <channels> <batch_size> <num_tot> <name_base> <activities>
```

For example:
```bash
python CSI_network.py ../data/ user1 340 100 1 16 4 model_v2 walking,sitting,standing,lying 
```

## Why This Update Was Necessary

The original architecture processed each antenna separately and then combined predictions, which:
1. Failed to capture correlations between antennas
2. Discarded 75% of the data when using a single-channel model
3. Led to poor accuracy and class bias (all predictions being class 5)

The new architecture processes all antennas simultaneously as channels, which:
1. Captures important cross-antenna correlations
2. Utilizes all available data
3. Significantly improves classification accuracy

## Verifying Your Model

After retraining, you can verify that your model has the correct architecture:

```bash
# Look for the line "Model expected input shape: (None, 340, 100, 4)"
python CSI_network_test.py <dir> <subdirs> <feature_length> <sample_length> <channels> <batch_size> <num_tot> <name_base> <activities>
```

## Troubleshooting

If you encounter the error message:
```
WARNING: Input shape (340, 100, 4) does not match model's expected shape (340, 100, 1)
```

This means you are using an old model trained with the single-channel architecture. Please retrain your model using the updated CSI_network.py script. 