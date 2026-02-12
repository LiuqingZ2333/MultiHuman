# Multi-Task Pose Dataset Tools

A comprehensive toolkit for generating multi-modal annotations and filtering pose datasets based on crowd density, pose complexity, and CLIP scores.

## Overview

This repository contains two main tools:

1. **MTdata_makelabel.py** - Multi-Annotation Generation Tool
2. **MTdata.py** - Multi-Dimensional Dataset Filtering Tool

## Features

### MTdata_makelabel.py

Automatically generates three types of annotations for image datasets:

- **Caption Generation**: Uses BLIP2 to generate image captions, ranked by CLIP scores
- **Depth Estimation**: Generates depth maps using depth-anything model
- **Pose Detection**: Detects human pose keypoints using MMPose (supports up to 20 people per image)

**Key Features:**
- Multi-GPU parallel processing support
- Batch processing of multiple datasets
- Selective annotation generation (enable/disable individual modules)
- Preserves directory structure in outputs
- Automatic skip of existing files for incremental processing

### MTdata.py

Filters pose datasets based on multiple criteria:

- **Crowd Density**: Filter by person density score (0-1 range)
- **Pose Complexity**: Filter by pose complexity score (0-1 range)  
- **CLIP Score**: Filter by image-text similarity score (0-1 range)

**Key Features:**
- Multi-dimensional filtering
- Statistical analysis and visualization
- Batch processing of multiple datasets
- Export filtered results to CSV
- Copy filtered images and pose files to new directory

## Requirements

```bash
# Core dependencies
torch>=1.10.0
numpy>=1.19.0
opencv-python>=4.5.0
tqdm>=4.60.0

# For Caption generation
transformers>=4.20.0
Pillow>=8.0.0

# For Pose detection
mmpose>=0.28.0
mmcv-full>=1.5.0

# For data handling
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pose-dataset-tools.git
cd pose-dataset-tools

# Install dependencies
pip install -r requirements.txt

# Install MMPose (if using pose detection)
pip install openmim
mim install mmcv-full
pip install mmpose
```

## Configuration

### MTdata_makelabel.py Configuration

Edit the `CONFIG` dictionary in the script:

```python
CONFIG = {
    "models": {
        "blip2": "/path/to/blip2",
        "clip": "/path/to/clip-vit-large-patch14",
        "depth": "/path/to/depth-anything",
        "mmpose_config": "/path/to/mmpose/config.py",
        "mmpose_checkpoint": "/path/to/mmpose/checkpoint.pth"
    },
    
    "datasets": {
        "Dataset1": {
            "image_dir": "/path/to/dataset1/images",
            "output_dir": "/path/to/dataset1/output"
        },
        "Dataset2": {
            "image_dir": "/path/to/dataset2/images",
            "output_dir": "/path/to/dataset2/output"
        }
    }
}
```

### MTdata.py Configuration

Edit the configuration dictionaries:

```python
# Dataset pose directories
DATASETS_CONFIG = {
    "Dataset1": "/path/to/dataset1/pose",
    "Dataset2": "/path/to/dataset2/pose"
}

# CLIP score CSV files (optional)
CLIP_CSV_CONFIG = {
    "Dataset1": "/path/to/dataset1/clip_scores.csv",
    "Dataset2": "/path/to/dataset2/clip_scores.csv"
}

# Filtering criteria
FILTER_CRITERIA = {
    "density": {"min": 0.3, "max": 1.0},
    "complexity": {"min": 0.4, "max": 1.0},
    "clip_score": {"min": 0.5, "max": 1.0}
}
```

## Usage

### MTdata_makelabel.py

**Generate all annotations for a single dataset:**
```bash
python MTdata_makelabel.py --datasets Dataset1 --enable-all --gpu 0
```

**Process multiple datasets:**
```bash
python MTdata_makelabel.py --datasets Dataset1,Dataset2 --enable-all --gpu 0
```

**Output Structure:**
```
output_dir/
├── captions/
│   ├── Dataset1_captions.json
│   └── Dataset1_captions.txt
├── depth/
│   └── [preserves original directory structure]
└── pose/
    └── [preserves original directory structure]
```

### MTdata.py

**Interactive mode:**
```bash
python MTdata.py
```

**Output Structure:**
```
filtered_data/
├── all_datasets_filtered_summary.csv
├── Dataset1_filtered_list.csv
├── Dataset1/
│   ├── images/
│   └── pose/
├── Dataset2_filtered_list.csv
└── Dataset2/
    ├── images/
    └── pose/
```


### Caption Output

**JSON format:**
```json
[
  {
    "image": "subdir/image1.jpg",
    "caption": "a person standing on a field",
    "score": 28.5432
  }
]
```

### Pose Output

Pose keypoints are saved as `.npz` files containing a numpy array with the following structure:
- 17 COCO keypoints per person
- Up to 20 people per image
- Each keypoint has (x, y, visibility) values

### Filtered Dataset CSV

```csv
Dataset,File Path,Person Count,Density,Complexity,CLIP Score
Dataset1,image1.npz,3,0.6521,0.7234,0.8123
Dataset1,image2.npz,5,0.7891,0.6543,0.7890
```

### Required Models

1. **BLIP2**: [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b)
2. **CLIP**: [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
3. **Depth-Anything**: [LiheYoung/depth-anything-large-hf](https://huggingface.co/LiheYoung/depth-anything-large-hf)
4. **MMPose**: [HigherHRNet checkpoint](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512.pth)


## License

This project is licensed under the MIT License - see the LICENSE file for details.
