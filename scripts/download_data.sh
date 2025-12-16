#!/bin/bash
# Download ESC-50 dataset from Kaggle

echo "Downloading ESC-50 dataset from Kaggle..."

# Make sure kaggle API is configured
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Error: Kaggle API credentials not found!"
    echo "Please download kaggle.json from https://www.kaggle.com/settings"
    echo "and place it in ~/.kaggle/kaggle.json"
    exit 1
fi

# Set correct permissions
chmod 600 ~/.kaggle/kaggle.json

# Create data directory if it doesn't exist
mkdir -p data/raw

# Download dataset
kaggle datasets download -d mmoreaux/environmental-sound-classification-50 -p data/raw

# Unzip dataset
cd data/raw
unzip -q environmental-sound-classification-50.zip
rm environmental-sound-classification-50.zip

echo "Dataset downloaded and extracted successfully!"
echo "Files are located in: data/raw/"
