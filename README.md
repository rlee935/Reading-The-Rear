# Reading-The-Rear

Reading-The-Rear investigates the privacy risks associated with high-resolution imagery captured by dashcams and autonomous vehicles. By utilizing YOLOv10, this project quantifies how much personal information—such as political views, religious beliefs, and family composition—is exposed through vehicle rear decals and stickers.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Workflow](#workflow)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Roadmap (Task Completion)](#roadmap)
- [License & Acknowledgments](#license--acknowledgments)

---

## Project Overview
These visual markers act as quasi-identifiers, potentially allowing for vehicle re-identification and tracking without license plate data. The project aims to evaluate these privacy "leaks" and test mitigations like blurring and digital noise to protect driver anonymity while maintaining camera utility.

## Workflow
1. **Data Collection:** Capture 5+ hours of public road footage via high-definition sensors.
2. **Automated Preprocessing:** Run YOLOv10 to detect vehicles, extract crops, and apply Gaussian blurring to license plates.
3. **Symbol Classification:** Train a custom classifier to identify political, religious, and family-related stickers.
4. **Baseline Evaluation:** Benchmark the automated results against a manually annotated dataset.
5. **Privacy Risk Assessment:** Evaluate the effectiveness of digital noise and image reduction in preserving anonymity.

---

## Getting Started

### Prerequisites
- **Python 3.12**: Required for compatibility with PyTorch CUDA wheels.
- **uv**: Recommended for high-speed dependency management and environment syncing.
- **NVIDIA GPU**: Required for local inference with YOLOv10 (CUDA 12.1+).

### Installation

#### Recommended: Using `uv`
```bash
# Clone the repository
git clone https://github.com/rlee935/Reading-The-Rear.git
cd Reading-The-Rear

# Sync the environment (automatically creates .venv and installs pinned dependencies)
uv sync
```

#### Standard: Using `pip`
```bash
# Clone the repository
git clone https://github.com/rlee935/Reading-The-Rear.git
cd Reading-The-Rear

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage
The primary entry point is `scripts/detect.py` for processing images and videos.

```bash
# Run detection and save cropped vehicle rears
uv run scripts/detect.py --source data/raw/{MEDIA_FILE} --save --show
```

---

## Project Structure
```text
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

---

## Model Architecture
This project implements **YOLOv10**, which removes the need for Non-Maximum Suppression (NMS), significantly reducing latency for real-time privacy filtering.

- **Backbone:** Efficient extraction of visual features from vehicle rears.
- **Privacy Layer:** Automated Gaussian blurring applied to detected license plate regions.

### Model Artifacts
- `yolov10n.pt`: Pre-trained YOLOv10 Nano weights. Used for initial vehicle detection and data filtering.

---

## Roadmap
### Phase 1: Infrastructure & Data (Mar 2 – Mar 22)
- [x] **Planning & Protocol**: Finalize the data collection plan and symbol taxonomies (Team: Mar 2 – Mar 8).  
- [x] **Environment Setup**: Configure uv environment with Python 3.12 and CUDA 12.1 (Raymond: Mar 2 – Mar 8).  
- [x] **Project Scaffolding**: Initialize GitHub repo with .gitignore and pyproject.toml (Raymond: Mar 2 – Mar 8).  
- [ ] **Data Acquisition**: Capture 5+ hours of MD-based highway/public road footage (Peter: Mar 9 – Mar 15).  
- [ ] **Detection Pipeline**: Implement YOLOv10 for automated vehicle detection and cropping (Raymond: Mar 16 – Mar 22).  
- [ ] **Anonymization**: Develop automated Gaussian blurring for license plates (Raymond: Mar 16 – Mar 22).  
- [ ] **Data Filtering**: Implement two-stage "gatekeeper" filtering to discard vehicles without decals (Peter: Mar 16 – Mar 22).  

### Phase 2: Annotation & Model Training (Mar 23 – Apr 5)
- [ ] **Manual Annotation**: Label 500+ stickers into categories for training (Adam: Mar 23 – Mar 29).  
- [ ] **Establish Baseline**: Coordinate human-annotator "Ground Truth" for model benchmarking (Vincent: Mar 23 – Mar 29).  
- [ ] **Classifier Development**: Train and tune custom neural network for symbol identification (Adam: Mar 30 – Apr 5).  
- [ ] **Midterm Delivery**: Complete 3-4 page progress report due March 31 (Team: Mar 30 – Apr 5).

### Phase 3: Mitigation & Analysis (Apr 6 – Apr 27)
- [ ] **Mitigation Testing**: Implement and evaluate digital noise and image reduction algorithms (Vincent: Apr 6 – Apr 12).  
- [ ] **Quantitative Analysis**: Calculate information exposure metrics and mitigation effectiveness (Vincent: Apr 13 – Apr 19).  
- [ ] **Final Synthesis**: Document findings in arXiv-style report and prepare final presentation (Team: Apr 20 – Apr 27).

---

## License & Acknowledgments
- **License**: This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
- **YOLOv10**: [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)
