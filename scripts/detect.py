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
    uv run scripts/detect.py --source data/raw/{FOOTAGE_FOLDER} --save --stride {FRAME_STRIDE} --workers {NUM_WORKERS}
"""

import argparse
import cv2
import os
import torch
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time

def get_file_hash(path):
    """Generate a simple hash based on mtime and size."""
    stats = os.stat(path)
    return hashlib.md5(f"{path}{stats.st_mtime}{stats.st_size}".encode()).hexdigest()

def load_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache, cache_path):
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=4)

def preprocess_frame(frame, n):
    """Placeholder for frame skipping logic."""
    pass

def process_source(source_path, vehicle_model, plate_model, output_dir, show=False, save=True, frame_stride=5):
    """Process a single video or image file."""
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        print(f"Error: Could not open source {source_path}")
        return 0

    frame_count = 0
    saved_count = 0
    
    # Get filename for unique saving if multiple videos
    source_stem = Path(source_path).stem
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_stride != 0:
            continue
            
        # Run vehicle detection
        results = vehicle_model.predict(frame, classes=[2, 5, 7], conf=0.5, verbose=False)[0]
        
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vw_full = x2 - x1
            vh_full = y2 - y1
            
            if vw_full < 80 or vh_full < 80:
                continue
                
            fh, fw_frame, _ = frame.shape
            pad_w, pad_h = int(vw_full * 0.1), int(vh_full * 0.1)
            px1_v, py1_v = max(0, x1 - pad_w), max(0, y1 - pad_h)
            px2_v, py2_v = min(fw_frame, x2 + pad_w), min(fh, y2 + pad_h)
            
            vehicle_crop = frame[py1_v:py2_v, px1_v:px2_v].copy()
            plate_results = plate_model.predict(vehicle_crop, conf=0.4, verbose=False)[0]
            
            found_plate = False
            max_p_conf = 0
            for p_box in plate_results.boxes:
                px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                p_conf = float(p_box.conf[0])
                max_p_conf = max(max_p_conf, p_conf)
                
                plate_roi = vehicle_crop[py1:py2, px1:px2]
                if plate_roi.size > 0:
                    blurred_plate = cv2.GaussianBlur(plate_roi, (51, 51), 0)
                    vehicle_crop[py1:py2, px1:px2] = blurred_plate
                    found_plate = True
            
            if found_plate:
                if save:
                    crop_name = f"{source_stem}_f{frame_count}_v{i}.jpg"
                    cv2.imwrite(os.path.join(output_dir, crop_name), vehicle_crop)
                    saved_count += 1
                print(f"  {source_stem} | Frame {frame_count} | Vehicle {i}: saved (Plate conf: {max_p_conf:.2f})")
        
        if show:
            # Draw detections for display
            display_frame = frame.copy()
            for box in results.boxes:
                dx1, dy1, dx2, dy2 = map(int, box.xyxy[0])
                cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
            cv2.imshow("Detections", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return saved_count
                
    cap.release()
    return saved_count

def detect_and_anonymize(source, weights, plate_weights, output_dir, show=False, save=True, frame_stride=5, workers=2, use_cache=True):
    """
    YOLOv10 Pipeline: Detect vehicles, use YOLOv8 sub-model to find and blur license plates, and crop rears.
    Supports directory sources, concurrency, and caching.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize models (shared across threads)
    vehicle_model = YOLO(weights).to(device)
    plate_model = YOLO(plate_weights).to(device)
    
    os.makedirs(output_dir, exist_ok=True)
    
    cache_path = os.path.join(output_dir, ".processing_cache.json")
    cache = load_cache(cache_path) if use_cache else {}
    
    source_path = Path(source)
    files_to_process = []
    
    if source_path.is_dir():
        extensions = ['*.mp4', '*.MP4', '*.avi', '*.mkv', '*.jpg', '*.png']
        for ext in extensions:
            files_to_process.extend(list(source_path.glob(ext)))
        files_to_process = sorted(files_to_process)
    else:
        files_to_process = [source_path]
    
    # Filter files using cache
    if use_cache:
        original_count = len(files_to_process)
        files_to_process = [f for f in files_to_process if get_file_hash(f) != cache.get(str(f))]
        skipped = original_count - len(files_to_process)
        if skipped > 0:
            print(f"Skipping {skipped} already processed files (cached).")
    
    if not files_to_process:
        print("No new files to process.")
        return

    print(f"Processing {len(files_to_process)} files with {workers} workers...")
    
    total_saved = 0
    start_time = time.time()
    
    import threading
    cache_lock = threading.Lock()
    
    def worker_func(file_path):
        nonlocal total_saved
        res = process_source(file_path, vehicle_model, plate_model, output_dir, show, save, frame_stride)
        if use_cache:
            with cache_lock:
                cache[str(file_path)] = get_file_hash(file_path)
                save_cache(cache, cache_path)
        return res

    if workers > 1 and len(files_to_process) > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(worker_func, files_to_process))
            total_saved = sum(results)
    else:
        for f in files_to_process:
            total_saved += worker_func(f)
            
    if use_cache:
        save_cache(cache, cache_path)
            
    elapsed = time.time() - start_time
    print(f"DONE. Finished in {elapsed:.2f}s. Total saved: {total_saved}")
    
    if show:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv10 Vehicle Detection & Anonymization")
    parser.add_argument("--source", type=str, required=True, help="Path to video or image")
    parser.add_argument("--weights", type=str, default="models/yolov10n.pt", help="Path to YOLOv10 weights")
    parser.add_argument("--plate-weights", type=str, default="models/plate_detector.pt", help="Path to plate sub-model weights")
    parser.add_argument("--output", type=str, default="data/processed/", help="Output directory")
    parser.add_argument("--save", action="store_true", help="Save crops")
    parser.add_argument("--show", action="store_true", help="Display stream")
    parser.add_argument("--stride", type=int, default=5, help="Frame skipping stride")
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent workers")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    
    args = parser.parse_args()
    
    detect_and_anonymize(
        args.source, 
        args.weights, 
        args.plate_weights,
        args.output, 
        show=args.show, 
        save=args.save, 
        frame_stride=args.stride,
        workers=args.workers,
        use_cache=not args.no_cache
    )