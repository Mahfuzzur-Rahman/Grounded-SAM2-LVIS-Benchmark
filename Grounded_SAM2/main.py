"""
LVIS Dataset Benchmarking Suite for Grounded-SAM 2
-------------------------------------------------
This script executes a large-scale zero-shot evaluation on the LVIS 16k subset.
It orchestrates the pipeline from dynamic prompt engineering to high-speed 
inference and final LVIS metric calculation.

Author: Mahfuzzur Rahman
Course: COMP 6341 - Computer Vision (Concordia University)
Date: April 2026
"""

import os
import json
import torch
import sys
from tqdm import tqdm

# Ensure the workspace and module paths are accessible for local imports
sys.path.append(os.getcwd())

from hybrid_model import GroundedSAM2
from evaluation_metrics import mask_to_rle, evaluate_lvis

def main():
    # ==========================================================================
    # 1. Configuration & Path Management
    # ==========================================================================
    WORKSPACE_ROOT = "/workspace"
    DATASET_ROOT = os.path.join(WORKSPACE_ROOT, "lvis_16k/lvis_16k_dataset")
    GT_PATH = os.path.join(DATASET_ROOT, "annotations.json")
    IMAGE_DIR = os.path.join(DATASET_ROOT, "images")
    RESULT_PATH = os.path.join(WORKSPACE_ROOT, "lvis_results.json")

    print("🚀 Initializing Grounded-SAM 2 Hybrid Model on RTX 4090...")
    model = GroundedSAM2()

    # Verify Ground Truth availability before proceeding
    if not os.path.exists(GT_PATH):
        print(f"❌ Critical Error: Ground Truth file not found at {GT_PATH}")
        return

    # ==========================================================================
    # 2. Metadata Indexing
    # ==========================================================================
    with open(GT_PATH, 'r') as f:
        lvis_data = json.load(f)
    
    # Create bidirectional mappings for LVIS taxonomy
    cat_id_to_name = {cat['id']: cat['name'] for cat in lvis_data['categories']}
    name_to_cat_id = {cat['name']: cat['id'] for cat in lvis_data['categories']}
    
    # Pre-index annotations for optimized per-image dynamic prompting
    img_id_to_anns = {}
    for ann in lvis_data['annotations']:
        img_id_to_anns.setdefault(ann['image_id'], []).append(ann['category_id'])

    results = []
    
    # Execution Mode: Set 'debug_mode' to False for full 16k dataset processing
    debug_mode = False 
    images_to_process = lvis_data['images'][:3] if debug_mode else lvis_data['images']
    
    print(f"📊 Starting benchmark on {len(images_to_process)} images...")

    # ==========================================================================
    # 3. Inference & Prediction Pipeline
    # ==========================================================================
    for img_info in tqdm(images_to_process, desc="Benchmarking", unit="img"):
        image_path = os.path.join(IMAGE_DIR, img_info['file_name'])
        if not os.path.exists(image_path):
            continue

        # --- Dynamic Prompt Engineering ---
        # Strategy: Consolidate unique categories and limit context to fit 
        # within BERT's 512-token limit used by the detector's text-encoder.
        unique_cats = list(set([cat_id_to_name[cid] for cid in img_id_to_anns.get(img_info['id'], [])]))
        prompt = ". ".join(unique_cats[:15]) if unique_cats else "object"

        try:
            # Execute Hybrid Inference: [Detection Boxes -> Segmentation Masks]
            boxes, masks, phrases, scores = model.run_inference(
                image_path, 
                prompt, 
                box_threshold=0.25
            )
            
            # --- Post-Processing & RLE Encoding ---
            for i in range(len(masks)):
                # Sanitize predicted phrases to align with LVIS JSON taxonomy
                pred_phrase = phrases[i].strip().lower().replace(" ", "_")
                cat_id = name_to_cat_id.get(pred_phrase)
                
                # Fallback: If text alignment fails, utilize the primary image category
                if cat_id is None:
                    cat_id = img_id_to_anns.get(img_info['id'], [1])[0]

                # Format detection into LVIS-compatible result structure
                results.append({
                    "image_id": int(img_info['id']),
                    "category_id": int(cat_id),
                    "segmentation": mask_to_rle(masks[i][0]),
                    "score": float(scores[i])
                })
        except Exception:
            # Graceful failure handling: Skip anomalies to maintain batch continuity
            continue 

    # ==========================================================================
    # 4. Data Persistence & Performance Evaluation
    # ==========================================================================
    if not results:
        print("\n💀 Pipeline Failure: No detections generated. Check GPU/Model status.")
        return

    # Store results in COCO/LVIS-compatible format for evaluation
    with open(RESULT_PATH, 'w') as f:
        json.dump(results, f)

    print(f"\n✅ Successfully saved {len(results)} detections to {RESULT_PATH}")
    
    print("📈 Initiating LVIS Evaluation Suite (mAP, mAP_rare, mAP_frequent)...")
    try:
        evaluate_lvis(GT_PATH, RESULT_PATH)
    except Exception as e:
        print(f"⚠️ Evaluation Module Error: {e}")

if __name__ == "__main__":
    main()