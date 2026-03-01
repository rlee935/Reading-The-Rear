# Reading-The-Rear

## Project Description

Reading-The-Rear investigates the privacy risks associated with high-resolution imagery captured by dashcams and autonomous vehicles . By utilizing YOLOv10, this project quantifies how much personal information—such as political views, religious beliefs, and family composition—is exposed through vehicle rear decals and stickers . These visual markers act as quasi-identifiers, potentially allowing for vehicle re-identification and tracking without license plate data .

The project aims to evaluate these privacy "leaks" and test mitigations like blurring and digital noise to protect driver anonymity while maintaining camera utility .

## Setup and Installation

### Prerequisites

- **Python 3.12**: Required for compatibility with PyTorch CUDA wheels.
- **uv**: Recommended for high-speed dependency management and environment syncing.
- **NVIDIA GPU**: Required for local inference with YOLOv10 (CUDA 12.1+).


### Installation (Recommended: `uv`)

```bash
# Clone the repository
git clone https://github.com/rlee935/Reading-The-Rear.git
cd Reading-The-Rear

# Sync the environment (automatically creates .venv and installs pinned dependencies)
uv sync
```

### Installation (Standard: `pip`)

```bash
# Clone the repository
git clone https://github.com/rlee935/Reading-The-Rear.git
cd Reading-The-Rear

# Create a virtual environment (recommended):
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies:
pip install -r requirements.txt
```

## Project Structure

```
Reading-The-Rear/
├── data/
│   ├── raw/            # Original dashcam footage
│   └── processed/      # Cropped vehicle rears for identification
├── models/
│   └── yolov10n.pt     # Pre-trained weights
├── scripts/            # Entry points for the pipeline
│   ├── detect.py       # Detection, cropping, and blurring
│   └── classify.py     # Custom symbol identification
├── pyproject.toml      # uv configuration
└── README.md
```

## Project Workflow
1. **Data Collection:** Capture 5+ hours of public road footage via high-definition sensors.
2. **Automated Preprocessing:** Run YOLOv10 to detect vehicles, extract crops, and apply Gaussian blurring to license plates .
3. **Symbol Classification:** Train a custom classifier to identify political, religious, and family-related stickers.
4. **Baseline Evaluation:** Benchmark the automated results against a manually annotated dataset.
5. **Privacy Risk Assessment:** Evaluate the effectiveness of digital noise and image reduction in preserving anonymity.

## Usage
The primary entry point is `scripts/detect.py` for processing images and videos.

```bash
# Run detection and save cropped vehicle rears
uv run scripts/detect.py --source <path_to_source> --save --show
```

## Model Architecture
This project implements YOLOv10, which removes the need for Non-Maximum Suppression (NMS), significantly reducing latency for real-time privacy filtering.
- **Backbone:** Efficient extraction of visual features from vehicle rears.
- **Privacy Layer:** Automated Gaussian blurring applied to detected license plate regions.

### Model Artifacts
- **yolov10n.pt**: Pre-trained YOLOv10 Nano weights. Used as the base model for 
  the "Main Experiment" to detect vehicles and discard non-relevant frames

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- YOLOv10: [https://github.com/THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)
