#!/usr/bin/env bash
# Script to recombine all split dataset files (.part_*) back to original files

set -e

echo "Searching for split files (.part_aa)..."
find . -type f -name "*.part_aa" | while read -r first_part; do
    original_file="${first_part%.part_aa}"
    echo "Recombining parts into '$original_file'..."
    cat "${original_file}.part_"* > "$original_file"
    echo "Recombined successfully: $original_file"
done

echo "All split files recombined successfully!"
