"""
Reading-The-Rear: detect.py
--------------------------
Lead: Raymond Lee (Machine Learning Engineer)
Task: Week 3 - Preprocessing and Dataset Refinement

Description:
    Implements the YOLOv10 pipeline to detect vehicles in raw dashcam footage.
    Automates the extraction of vehicle rear crops and applies Gaussian blurring 
    to license plate regions to ensure privacy-compliant data handling.

Usage:
    uv run scripts/detect.py --source data/raw/video.mp4 --save
"""