#!/bin/bash

# Get the current directory
current_path="$(pwd)"

# Normalize Windows-style UNC path (if mounted)
# For normal local paths this is fine as-is
safe_path="$current_path"

# Add the directory to Git's global safe.directory list
git config --global --add safe.directory "$safe_path"

echo "Marked as safe Git directory: $safe_path"

# Usage
# Save this script as mark-git-safe.sh
# Make it executable: chmod +x mark-git-safe.sh
# Run it in the directory you want to mark as safe: ./mark-git-safe.sh
