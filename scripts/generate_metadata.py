#!/usr/bin/env python3
"""
Generate experiment metadata files for all experiment directories.
This script creates standardized meta.json files in each experiment directory
to provide consistent metadata tracking across all research questions.
"""

import os
import json
import argparse
import glob
import datetime
import re
from pathlib import Path


def extract_domains_from_id(experiment_id):
    """
    Extract source and target domains based on experiment ID
    """
    all_domains = ["bedroom", "living", "kitchen", "lab", "office", "semi"]
    
    if experiment_id.startswith("no_"):
        # This is a leave-one-out experiment
        target_domain = experiment_id.replace("no_", "")
        source_domains = [d for d in all_domains if d != target_domain]
        return source_domains, [target_domain]
    elif experiment_id.startswith("source"):
        # This is a source scaling experiment
        try:
            # Extract just the numeric part
            source_str = experiment_id.replace("source", "")
            # Remove any trailing characters (like _4)
            num_sources = int(''.join(c for c in source_str if c.isdigit()))
            
            if num_sources == 1:
                return ["bedroom_split1"], ["bedroom_split2"]
            elif num_sources == 2:
                return ["bedroom_split1", "living_split1"], ["bedroom_split2", "living_split2"]
            elif num_sources == 3:
                return ["bedroom_split1", "living_split1", "office_split1"], ["bedroom_split2", "living_split2", "office_split2"]
            elif num_sources == 4:
                return ["bedroom_split1", "living_split1", "office_split1", "semi_split1"], ["bedroom_split2", "living_split2", "office_split2", "semi_split2"]
        except ValueError:
            print(f"Warning: Could not parse source count from '{experiment_id}', using default domains")
    
    # Default fallback
    return ["unknown"], ["unknown"]


def determine_components(experiment_id):
    """
    Determine which components are used based on experiment ID
    """
    # Default: all components are used
    components = ["phase_sanitization", "antenna_randomization"]
    
    if "no_phase" in experiment_id:
        components.remove("phase_sanitization")
    if "no_antenna" in experiment_id:
        components.remove("antenna_randomization")
        
    return components


def generate_metadata_for_experiment(exp_dir, dry_run=False):
    """
    Generate metadata file for a single experiment directory
    """
    # Determine research question based on directory path
    if "RQ1_generalization" in exp_dir:
        research_question = "RQ1"
    elif "RQ2_technique_comparison" in exp_dir:
        research_question = "RQ2"
    elif "RQ3_tradeoffs" in exp_dir:
        research_question = "RQ3"
    elif "RQ4_source_scaling" in exp_dir:
        research_question = "RQ4"
    elif "RQ5_component_analysis" in exp_dir:
        research_question = "RQ5"
    else:
        research_question = "unknown"
    
    # Extract experiment ID from directory name
    experiment_id = os.path.basename(exp_dir)
    
    # Determine source and target domains
    source_domains, target_domains = extract_domains_from_id(experiment_id)
    
    # Determine components used
    components_used = determine_components(experiment_id)
    
    # Create metadata object
    metadata = {
        "research_question": research_question,
        "experiment_id": experiment_id,
        "train_domains": source_domains,
        "test_domains": target_domains,
        "components_used": components_used,
        "timestamp": datetime.datetime.now().isoformat(timespec='seconds')
    }
    
    # Create the metadata file path
    metadata_file = os.path.join(exp_dir, "meta.json")
    
    if dry_run:
        print(f"Would create {metadata_file}:")
        print(json.dumps(metadata, indent=4))
        return True
    
    try:
        # Write metadata to file
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Created metadata file: {metadata_file}")
        return True
    except Exception as e:
        print(f"Error creating metadata file for {exp_dir}: {e}")
        return False


def generate_all_metadata(results_dir, dry_run=False):
    """
    Generate metadata files for all experiment directories
    """
    success_count = 0
    failure_count = 0
    
    # RQ1: Leave-one-out experiments
    rq1_dirs = glob.glob(os.path.join(results_dir, "RQ1_generalization/leave_one_out/no_*"))
    for exp_dir in rq1_dirs:
        if os.path.isdir(exp_dir):
            if generate_metadata_for_experiment(exp_dir, dry_run):
                success_count += 1
            else:
                failure_count += 1
    
    # RQ4: Source scaling experiments
    rq4_dirs = glob.glob(os.path.join(results_dir, "RQ4_source_scaling/source_*"))
    for exp_dir in rq4_dirs:
        if os.path.isdir(exp_dir):
            if generate_metadata_for_experiment(exp_dir, dry_run):
                success_count += 1
            else:
                failure_count += 1
    
    # RQ5: Component analysis experiments
    rq5_dirs = glob.glob(os.path.join(results_dir, "RQ5_component_analysis/no_*"))
    for exp_dir in rq5_dirs:
        if os.path.isdir(exp_dir) and not exp_dir.endswith("_plots"):
            if generate_metadata_for_experiment(exp_dir, dry_run):
                success_count += 1
            else:
                failure_count += 1
    
    print(f"\nMetadata generation summary:")
    print(f"  Successfully processed: {success_count} directories")
    print(f"  Failures: {failure_count} directories")
    
    if dry_run:
        print("\nThis was a dry run. No files were actually created.")


def main():
    parser = argparse.ArgumentParser(description="Generate metadata files for experiment directories")
    parser.add_argument("--results-dir", default="./results", help="Root directory containing experiment results")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without creating files")
    args = parser.parse_args()
    
    generate_all_metadata(args.results_dir, args.dry_run)
    
    return 0


if __name__ == "__main__":
    exit(main()) 