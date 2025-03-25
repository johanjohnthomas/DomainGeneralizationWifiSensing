#!/usr/bin/env python3

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys

def load_history(history_file):
    """Load training history from pickle file"""
    try:
        with open(history_file, 'rb') as f:
            history = pickle.load(f)
        return history
    except Exception as e:
        print(f"Error loading history file: {e}")
        sys.exit(1)

def plot_accuracy(history, output_dir, model_name):
    """Plot training and validation accuracy curves"""
    plt.figure(figsize=(10, 6))
    
    # Determine which accuracy key is in the history
    acc_key = None
    val_acc_key = None
    
    if 'sparse_categorical_accuracy' in history:
        acc_key = 'sparse_categorical_accuracy'
        val_acc_key = 'val_sparse_categorical_accuracy'
    elif 'accuracy' in history:
        acc_key = 'accuracy'
        val_acc_key = 'val_accuracy'
    elif 'acc' in history:
        acc_key = 'acc'
        val_acc_key = 'val_acc'
    else:
        print("No accuracy metric found in history.")
        return
    
    # Plot training accuracy
    plt.plot(history[acc_key], label='Training Accuracy')
    
    # Plot validation accuracy if available
    if val_acc_key in history:
        plt.plot(history[val_acc_key], label='Validation Accuracy')
    
    plt.title(f'Training Accuracy - {model_name}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Save the plot
    output_file = os.path.join(output_dir, f'{model_name}_accuracy.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Accuracy plot saved to {output_file}")

def plot_loss(history, output_dir, model_name):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    plt.plot(history['loss'], label='Training Loss')
    
    # Plot validation loss if available
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    
    plt.title(f'Training Loss - {model_name}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Save the plot
    output_file = os.path.join(output_dir, f'{model_name}_loss.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {output_file}")

def plot_combined_metrics(history, output_dir, model_name):
    """Plot all metrics in a single figure with subplots"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Determine which accuracy key is in the history
    acc_key = None
    val_acc_key = None
    
    if 'sparse_categorical_accuracy' in history:
        acc_key = 'sparse_categorical_accuracy'
        val_acc_key = 'val_sparse_categorical_accuracy'
    elif 'accuracy' in history:
        acc_key = 'accuracy'
        val_acc_key = 'val_accuracy'
    elif 'acc' in history:
        acc_key = 'acc'
        val_acc_key = 'val_acc'
    
    # Plot accuracy metrics
    if acc_key is not None:
        ax1.plot(history[acc_key], 'b-', label='Training Accuracy')
        if val_acc_key in history:
            ax1.plot(history[val_acc_key], 'g-', label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.legend(loc='lower right')
        ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot loss metrics
    ax2.plot(history['loss'], 'r-', label='Training Loss')
    if 'val_loss' in history:
        ax2.plot(history['val_loss'], 'm-', label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Use integer x-axis
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, f'{model_name}_training_curves.png')
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Combined training curves saved to {output_file}")

def create_metrics_table(history, output_dir, model_name):
    """Create a text file with summary metrics from the training history"""
    metrics_file = os.path.join(output_dir, f'{model_name}_training_metrics.txt')
    
    with open(metrics_file, 'w') as f:
        f.write(f"Training Metrics Summary for {model_name}\n")
        f.write("="*50 + "\n\n")
        
        # Determine which accuracy key is in the history
        acc_key = None
        if 'sparse_categorical_accuracy' in history:
            acc_key = 'sparse_categorical_accuracy'
        elif 'accuracy' in history:
            acc_key = 'accuracy'
        elif 'acc' in history:
            acc_key = 'acc'
            
        # Write metric values
        if acc_key is not None:
            max_acc = max(history[acc_key])
            final_acc = history[acc_key][-1]
            f.write(f"Maximum Training Accuracy: {max_acc:.4f}\n")
            f.write(f"Final Training Accuracy: {final_acc:.4f}\n")
        
        min_loss = min(history['loss'])
        final_loss = history['loss'][-1]
        f.write(f"Minimum Training Loss: {min_loss:.4f}\n")
        f.write(f"Final Training Loss: {final_loss:.4f}\n")
        
        # Number of epochs
        num_epochs = len(history['loss'])
        f.write(f"\nTotal Epochs: {num_epochs}\n")
        
        # Early stopping info if the training didn't use all epochs
        f.write("\nNote: If the number of epochs is less than expected,\n")
        f.write("early stopping might have been used during training.\n")
    
    print(f"Training metrics summary saved to {metrics_file}")

def main():
    parser = argparse.ArgumentParser(description='Plot training curves from model history')
    parser.add_argument('--history', type=str, help='Path to history pickle file',
                        default='./models/no_bedroom_E_J_L_R_W_history.pkl')
    parser.add_argument('--output-dir', type=str, help='Directory to save plots',
                        default='./results/RQ1_generalization/leave_one_out/no_bedroom/plots')
    parser.add_argument('--model-name', type=str, help='Model name for plot titles',
                        default='no_bedroom')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract the model name from the history filename if not specified
    if args.model_name == 'no_bedroom' and os.path.exists(args.history):
        base_name = os.path.basename(args.history)
        model_name = os.path.splitext(base_name)[0].replace('_history', '')
        args.model_name = model_name
    
    print(f"Loading history from: {args.history}")
    history = load_history(args.history)
    
    # Print available keys in history
    print("Available metrics in history:", list(history.keys()))
    
    # Generate plots
    plot_accuracy(history, args.output_dir, args.model_name)
    plot_loss(history, args.output_dir, args.model_name)
    plot_combined_metrics(history, args.output_dir, args.model_name)
    create_metrics_table(history, args.output_dir, args.model_name)
    
    print("Training visualization complete!")

if __name__ == "__main__":
    main() 