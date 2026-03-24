"""
Reading-The-Rear: 1_detect.py
--------------------------
Lead: Raymond Lee (Machine Learning Engineer)
Task: Week 3 - Preprocessing and Dataset Refinement

Description:
    Implements the YOLOv10 pipeline to detect vehicles in raw dashcam footage.
    Automates the extraction of vehicle rear crops and applies Gaussian blurring 
    to license plate regions to ensure privacy-compliant data handling.

Usage:
    uv run scripts/1_detect.py --source data/raw/ --save --stride 30 --workers 8
"""
import argparse
import os
import torch
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import json
import hashlib
import time
from tqdm import tqdm
import multiprocessing as mp
import traceback
import cv2
import queue

# --- Utility Functions ---

def get_file_hash(path):
    """Generates a simple hash based on file stats to track changes."""
    stats = os.stat(path)
    return hashlib.md5(f"{path}{stats.st_mtime}{stats.st_size}".encode()).hexdigest()

def load_cache(cache_path):
    """Loads the processing cache from a JSON file."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except: return {}
    return {}

def save_cache(cache, cache_path):
    """Saves the processing cache to a JSON file."""
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=4)

FILE_DONE = 'FILE_DONE'
WORKER_DONE = 'WORKER_DONE'

def video_decoder_worker(files_chunk, stride, frame_queue, error_queue):
    """Worker Process: Decodes video frames and pushes them to the frame queue."""
    cv2.setNumThreads(0)
    
    for source_path in files_chunk:
        source_stem = Path(source_path).stem
        try:
            cap = cv2.VideoCapture(str(source_path))
            if not cap.isOpened():
                frame_queue.put(FILE_DONE)
                continue
            
            frame_idx = 0
            while True:
                # Only process every Nth frame (stride)
                if frame_idx % stride == 0:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    frame_queue.put((source_stem, frame_idx + 1, frame))
                else:
                    ret = cap.grab() # Faster than read() when skipping frames
                    if not ret:
                        break
                frame_idx += 1
            
            cap.release()
                    
        except Exception as e:
            error_queue.put(f"{source_path}: {e}")
        
        frame_queue.put(FILE_DONE)
            
    frame_queue.put(WORKER_DONE)

def async_writer_worker(write_queue):
    """Worker Process: Handles file I/O operations in a separate process."""
    while True:
        item = write_queue.get()
        if item is None: break
        path, img = item
        cv2.imwrite(path, img)

def gpu_inference_loop(frame_queue, write_queue, active_decoders, weights, plate_weights, output_dir, save, show, total_files, batch_size=16, total_frames=None):
    """Main GPU loop: Orchestrates detection models and processes frame batches."""
    try:
        # Load YOLO models
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print(f"Loading TensorRT/CUDA engines on {device} (Batch Size: {batch_size})...")
        else:
            print(f"Loading YOLO models on {device} (Batch Size: {batch_size})...")
            
        v_model = YOLO(weights, task='detect')
        p_model = YOLO(plate_weights, task='detect')

        saved_count = 0
        frames_processed = 0
        files_completed = 0
        workers_running = active_decoders
        
        # Initialize progress bar
        pbar = tqdm(total=total_frames, desc="Processing", unit="frame",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} frames [{elapsed}<{remaining}, {rate_fmt}] | {postfix}')
        pbar.set_postfix(files=f"0/{total_files}", crops=0)
        
        batch_frames = []
        batch_meta = []

        def process_batch(frames, metas):
            """Internal helper to run inference on a batch of frames."""
            nonlocal saved_count
            if not frames: return
            
            # 1. Detect Vehicles (Cars, Buses, Trucks)
            use_half = (device == 'cuda')
            v_results = v_model.predict(frames, classes=[2, 5, 7], conf=0.5, verbose=False, half=use_half, batch=len(frames))
            
            all_v_crops = []
            all_v_meta = []

            for f_idx, res in enumerate(v_results):
                if not res.boxes: continue
                
                frame = frames[f_idx]
                source_stem = metas[f_idx]['stem']
                frame_count = metas[f_idx]['fcount']
                fh, fw, _ = frame.shape

                for i, box in enumerate(res.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vw, vh = x2 - x1, y2 - y1
                    if vw < 80 or vh < 80: continue # Skip small detections
                    
                    # Add 10% padding for better context
                    pw, ph = int(vw * 0.1), int(vh * 0.1)
                    px1, py1 = max(0, x1 - pw), max(0, y1 - ph)
                    px2, py2 = min(fw, x2 + pw), min(fh, y2 + ph)
                    
                    v_crop = frame[py1:py2, px1:px2].copy()
                    all_v_crops.append(v_crop)
                    all_v_meta.append({'stem': source_stem, 'fcount': frame_count, 'id': i})

            # 2. Detect and Blur License Plates on Vehicle Crops
            if all_v_crops:
                p_results = p_model.predict(all_v_crops, conf=0.4, verbose=False, half=use_half, batch=len(all_v_crops))
                
                for idx, p_res in enumerate(p_results):
                    target_img = all_v_crops[idx]
                    
                    # Apply Gaussian Blur to any detected plates for privacy
                    for p_box in p_res.boxes:
                        bx1, by1, bx2, by2 = map(int, p_box.xyxy[0])
                        roi = target_img[by1:by2, bx1:bx2]
                        if roi.size > 0:
                            target_img[by1:by2, bx1:bx2] = cv2.GaussianBlur(roi, (51, 51), 0)
                    
                    # Queue the blurred image for saving
                    if save:
                        meta = all_v_meta[idx]
                        crop_name = f"{meta['stem']}_f{meta['fcount']}_v{meta['id']}.jpg"
                        write_queue.put((os.path.join(output_dir, crop_name), target_img))
                        saved_count += 1

        # Main processing loop: Collect frames into batches
        while workers_running > 0 or not frame_queue.empty():
            try:
                item = frame_queue.get(timeout=0.1)
            except queue.Empty:
                if batch_frames:
                    process_batch(batch_frames, batch_meta)
                    batch_frames, batch_meta = [], []
                continue
                
            if item == WORKER_DONE:
                workers_running -= 1
                continue
            
            if item == FILE_DONE:
                if batch_frames:
                    process_batch(batch_frames, batch_meta)
                    batch_frames, batch_meta = [], []
                files_completed += 1
                pbar.set_postfix(files=f"{files_completed}/{total_files}", crops=saved_count)
                continue
                
            source_stem, frame_count, frame = item
            frames_processed += 1
            batch_frames.append(frame)
            batch_meta.append({'stem': source_stem, 'fcount': frame_count})

            # Run inference when batch size is reached
            if len(batch_frames) >= batch_size:
                process_batch(batch_frames, batch_meta)
                pbar.update(len(batch_frames))
                batch_frames, batch_meta = [], []
                pbar.set_postfix(files=f"{files_completed}/{total_files}", crops=saved_count)
            
            if show:
                cv2.imshow("Processing", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        # Flush any remaining frames in the final batch
        if batch_frames:
            process_batch(batch_frames, batch_meta)
            pbar.update(len(batch_frames))
        pbar.close()
        return saved_count
    except Exception as e:
        print(f"Inference error: {e}")
        traceback.print_exc()
        return 0

def main():
    # Setup CLI Arguments
    parser = argparse.ArgumentParser(description="Batch Optimized YOLO Pipeline")
    parser.add_argument("--source", type=str, required=True, help="Path to video/image folder")
    parser.add_argument("--weights", type=str, default="models/yolov10n.pt", help="Path to .pt vehicle model")
    parser.add_argument("--plate-weights", type=str, default="models/plate_detector.pt", help="Path to .pt plate model")
    parser.add_argument("--output", type=str, default="data/1_license_plate/")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel CPUs for video decoding")
    parser.add_argument("--batch", type=int, default=16, help="GPU Batch Size")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    # Initialize output directory and cache
    os.makedirs(args.output, exist_ok=True)
    cache_path = os.path.join(args.output, ".processing_cache.json")
    cache = load_cache(cache_path) if not args.no_cache else {}

    # Discover files to process
    source_p = Path(args.source)
    all_files = []
    
    if source_p.is_dir():
        exts = {'.mp4', '.MP4', '.avi', '.mkv', '.jpg', '.png'}
        for root, dirs, files in os.walk(str(source_p), followlinks=True):
            for file in files:
                if Path(file).suffix in exts:
                    all_files.append(Path(root) / file)
    else:
        all_files = [source_p]
    
    # Filter files based on cache
    files_to_run = []
    if not args.no_cache:
        for f in all_files:
            if cache.get(str(f)) != get_file_hash(f):
                files_to_run.append(f)
        print(f"Skipping {len(all_files) - len(files_to_run)} cached files.")
    else:
        files_to_run = all_files

    if not files_to_run:
        print("No new files to process.")
        return

    print(f"Processing {len(files_to_run)} files...")
    
    # Estimate total frames for progress bar
    total_frames = 0
    for f in files_to_run:
        cap = cv2.VideoCapture(str(f))
        total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // args.stride
        cap.release()

    start_time = time.time()

    # Setup Multiprocessing
    mp.set_start_method('spawn', force=True)
    frame_queue = mp.Queue(maxsize=128)
    write_queue = mp.Queue()
    error_queue = mp.Queue()

    # Launch Decoder Processes
    actual_workers = min(args.workers, len(files_to_run))
    file_chunks = np.array_split(files_to_run, actual_workers)
    
    print(f"Starting {actual_workers} CPU decoding workers...")
    
    decoder_processes = []
    for chunk in file_chunks:
        if len(chunk) > 0:
            p = mp.Process(target=video_decoder_worker, args=(chunk, args.stride, frame_queue, error_queue))
            p.start()
            decoder_processes.append(p)

    # Launch Writer Process
    writer_process = mp.Process(target=async_writer_worker, args=(write_queue,))
    writer_process.start()
    
    try:
        # Run GPU inference (main loop)
        total_saved = gpu_inference_loop(
            frame_queue, write_queue, len(decoder_processes), 
            args.weights, args.plate_weights, args.output, args.save, args.show,
            total_files=len(files_to_run),
            batch_size=args.batch,
            total_frames=total_frames
        )
            
        # Update cache after successful processing
        for f in files_to_run:
            if not args.no_cache:
                cache[str(f)] = get_file_hash(f)
        
        if not args.no_cache:
            save_cache(cache, cache_path)
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saved cache.")
    except Exception as e:
        print(f"\nExecution failed: {e}")
        traceback.print_exc()
        
    # Final cleanup of processes
    for p in decoder_processes:
        p.join()
        
    write_queue.put(None)
    writer_process.join()

    while not error_queue.empty():
        print(f"Worker Error: {error_queue.get()}")

    elapsed = time.time() - start_time
    print(f"\nCOMPLETED | Saved: {total_saved} crops | Elapsed: {elapsed:.2f}s | Avg: {(elapsed/len(files_to_run)):.2f}s/file")
    if args.show: cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
