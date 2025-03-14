#!/bin/bash

# Clear output file at the start
: > output.txt

# Loop through Python files only, excluding files with 'plot' or 'test' in the name
find . -type f -name "*.py" -not -name "*plot*" -not -name "*test*" -print0 | while IFS= read -r -d '' file; do
    echo "$file:" >> output.txt
    echo '```' >> output.txt
    cat "$file" >> output.txt
    echo '```' >> output.txt
    echo "" >> output.txt
done

# Export the Makefile if it exists
if [ -f ./Makefile ]; then
    echo "./Makefile:" >> output.txt
    echo '```' >> output.txt
    cat ./Makefile >> output.txt
    echo '```' >> output.txt
    echo "" >> output.txt
fi