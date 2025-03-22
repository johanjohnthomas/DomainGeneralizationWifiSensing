#!/usr/bin/env python3
"""
Script to fix device placement issues in CSI_network.py
"""

import os
import sys

def fix_csi_network():
    """
    Add the necessary code to CSI_network.py to handle GPU device placement properly
    """
    with open('CSI_network.py', 'r') as file:
        content = file.read()
    
    # Check if the fix has already been applied
    if "# GPU FIX APPLIED" in content:
        print("Fix already applied to CSI_network.py")
        return
    
    # Add GPU configuration after the imports and before the main code
    gpu_config = """
# GPU FIX APPLIED
# GPU device placement handling
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# Fix the device placement issue
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Set memory growth for all GPUs
        for dev in physical_devices:
            tf.config.experimental.set_memory_growth(dev, True)
            print(f"Memory growth enabled for {dev}")
        
        # Make TensorFlow use the first GPU only
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        
        # Force operations to be on the same device
        tf.config.set_soft_device_placement(True)
        
        # Disable XLA compilation which can sometimes cause device placement issues
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
        
        print("GPU configuration complete")
    except Exception as e:
        print(f"GPU configuration error: {e}")

"""
    
    # Find the right spot to insert the GPU config (after imports, before main)
    import_section_end = content.find("if __name__ == '__main__':")
    if import_section_end == -1:
        print("Could not find the main section in CSI_network.py")
        return
    
    # Insert the GPU config
    new_content = content[:import_section_end] + gpu_config + content[import_section_end:]
    
    # We'll take a different approach to modify the prediction section
    # Read the file line by line to handle indentation properly
    lines = new_content.split('\n')
    modified_lines = []
    
    i = 0
    while i < len(lines):
        # Add the current line
        modified_lines.append(lines[i])
        
        # Check if this is the prediction line
        if "train_prediction_list = csi_model.predict" in lines[i]:
            # Find indentation of the current line
            current_indent = len(lines[i]) - len(lines[i].lstrip())
            
            # Insert the CPU context before this line with the same indentation
            cpu_line = " " * current_indent + "# Move model to CPU for prediction to avoid device placement issues"
            with_line = " " * current_indent + "with tf.device('/CPU:0'):"
            
            # Replace the current line with the CPU context
            modified_lines[-1] = cpu_line
            modified_lines.append(with_line)
            
            # Add the prediction line with increased indentation
            modified_lines.append(" " * (current_indent + 4) + lines[i].lstrip())
            
            # Skip the original prediction line since we've already added it
            i += 1
            
            # Increase indentation for all lines until we hit a line with the same or less indentation
            while i < len(lines):
                next_line = lines[i]
                
                # If the line is empty, add it without changes
                if not next_line.strip():
                    modified_lines.append(next_line)
                    i += 1
                    continue
                
                # Calculate the indentation of the next line
                next_indent = len(next_line) - len(next_line.lstrip())
                
                # If indentation is less than or equal to the original line, we're out of the block
                if next_indent <= current_indent:
                    break
                
                # Otherwise, add the line with increased indentation
                modified_lines.append(" " * 4 + next_line)
                i += 1
        else:
            i += 1
    
    # Add any remaining lines
    while i < len(lines):
        modified_lines.append(lines[i])
        i += 1
    
    # Join the lines back into content
    new_content = '\n'.join(modified_lines)
    
    # Write the modified content back
    with open('CSI_network.py.new', 'w') as file:
        file.write(new_content)
    
    # Create a backup of the original file
    if os.path.exists('CSI_network.py.bak'):
        os.remove('CSI_network.py.bak')
    os.rename('CSI_network.py', 'CSI_network.py.bak')
    
    # Replace the original with the modified version
    os.rename('CSI_network.py.new', 'CSI_network.py')
    
    print("Successfully applied fix to CSI_network.py")
    print("Original file backed up as CSI_network.py.bak")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    fix_csi_network() 