#!/usr/bin/env python
"""
This script filters activity directories to:
1. Only keep activities specified by the user (default: E, J, L, W)
2. Only keep one instance of each activity type per domain

You can specify which activities to include using the --activities parameter.
"""

import os
import glob
import argparse
import re
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('doppler_dir', help='Base directory containing AR directories')
    parser.add_argument('--activities', default='E,J,L,W', help='Comma-separated list of allowed activities')
    parser.add_argument('--domains', help='Comma-separated list of domains to process. If not specified, process all.')
    parser.add_argument('--output', default='filtered_activities.txt', help='Output file for filtered directories')
    args = parser.parse_args()
    
    # Parse allowed activities
    allowed_activities = set(args.activities.split(','))
    print(f"Filtering for activities: {', '.join(sorted(allowed_activities))}")
    
    # Get domains to process
    if args.domains:
        domains = args.domains.split(',')
    else:
        # Find all AR directories
        domains = []
        for item in os.listdir(args.doppler_dir):
            item_path = os.path.join(args.doppler_dir, item)
            if os.path.isdir(item_path) and item.startswith('AR'):
                domains.append(item)
    
    print(f"Processing domains: {', '.join(domains)}")
    
    # Dictionary to store the selected activity directories for each domain
    selected_dirs = {}
    
    # Process each domain
    for domain in domains:
        domain_path = os.path.join(args.doppler_dir, domain)
        if not os.path.isdir(domain_path):
            print(f"Warning: Domain directory not found: {domain_path}")
            continue
        
        # Dictionary to store activities found in this domain
        domain_activities = defaultdict(list)
        
        # Find all activity directories
        activity_pattern = re.compile(f'{domain}_(\\w+)')
        for item in os.listdir(domain_path):
            item_path = os.path.join(domain_path, item)
            if not os.path.isdir(item_path):
                continue
                
            match = activity_pattern.match(item)
            if not match:
                continue
                
            activity_with_num = match.group(1)
            # Extract base activity (first letter)
            base_activity = activity_with_num[0]
            
            # Skip if not in allowed activities
            if base_activity not in allowed_activities:
                continue
            
            # Store activity directory with its relative path (just domain/dir_name)
            # This prevents path duplication issues
            domain_activities[base_activity].append(os.path.join(domain, item))
        
        # Select one instance of each activity type
        domain_selected = {}
        for activity, dirs in domain_activities.items():
            # Sort directories so we always select the same one (e.g., E1 before E2)
            dirs.sort()
            if dirs:
                # Select first directory for each activity type
                domain_selected[activity] = dirs[0]
                print(f"Selected for {domain} activity {activity}: {os.path.basename(dirs[0])}")
        
        selected_dirs[domain] = domain_selected
    
    # Count total selected directories
    total_selected = sum(len(activities) for activities in selected_dirs.values())
    print(f"Total selected activity directories: {total_selected}")
    
    # Write selected directories to output file
    with open(args.output, 'w') as f:
        for domain, activities in selected_dirs.items():
            for activity, dir_path in activities.items():
                f.write(f"{dir_path}\n")
    
    print(f"Filtered activity directories written to: {args.output}")
    
    # Also generate an easy-to-parse summary
    summary_file = args.output + '.summary'
    with open(summary_file, 'w') as f:
        for domain, activities in selected_dirs.items():
            f.write(f"{domain}:")
            for activity in sorted(activities.keys()):
                f.write(f" {activity}")
            f.write("\n")
    
    print(f"Summary written to: {summary_file}")
    
    return 0

if __name__ == '__main__':
    exit(main()) 