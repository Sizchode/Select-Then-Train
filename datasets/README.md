# Datasets

This directory contains datasets used in the STT project.

## Available Datasets

### CLUTRR
- **Location**: `clutrr/` (included in repository)
- **Size**: ~3MB
- **Format**: JSON files (train.json, val.json, test.json)
- **Status**: ✅ Included in repository

### EuroSAT
- **Location**: `eurosat.zip` (local only, not in repository)
- **Size**: ~91MB (compressed)
- **Format**: ZIP archive
- **Status**: ⚠️ Too large for GitHub (exceeds 50MB limit)
- **How to get**: Download from [EuroSAT dataset](https://github.com/phelber/EuroSAT) or extract from your local copy

### GTSRB (German Traffic Sign Recognition Benchmark)
- **Location**: `gtsrb.zip` (local only, not in repository)
- **Size**: ~267MB (compressed)
- **Format**: ZIP archive
- **Status**: ⚠️ Too large for GitHub (exceeds 100MB limit)
- **How to get**: Download from [GTSRB official website](https://benchmark.ini.rub.de/gtsrb_dataset.html) or extract from your local copy

## Usage

For datasets included in the repository (CLUTRR), they are automatically available.

For large datasets (EuroSAT, GTSRB), you need to:

1. **Option 1**: Download the datasets from their official sources
2. **Option 2**: If you have access to the original data location, extract the zip files:
   ```bash
   cd datasets/
   unzip eurosat.zip
   unzip gtsrb.zip
   ```

3. **Option 3**: Update the paths in `stt/dataset/classification/get_loader.py` to point to your dataset locations

## Notes

- Large dataset files (`.zip`) are excluded from git via `.gitignore`
- Update dataset paths in the code to match your local setup
- See `stt/dataset/classification/get_loader.py` for dataset loading configuration

