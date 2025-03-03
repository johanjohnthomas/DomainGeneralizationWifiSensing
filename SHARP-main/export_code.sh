#!/bin/bash

# Clear output file at the start
: > output.txt

# Loop through Python files only
find . -type f -name "*.py" -print0 | while IFS= read -r -d '' file; do
    echo "$file:" >> output.txt
    echo '```' >> output.txt
    cat "$file" >> output.txt
    echo '```' >> output.txt
    echo "" >> output.txt
done