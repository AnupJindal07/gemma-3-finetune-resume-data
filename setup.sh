#!/bin/bash
echo "Setting up Resume Analyzer AI..."

# Create necessary directories
mkdir -p data/raw_data data/processed_data models/checkpoints

# Install dependencies
#pip install -r requirements.txt
pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt

echo "Setup complete!"