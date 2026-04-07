"""
LVIS Evaluation Metrics Module
------------------------------
Handles the transformation of binary segmentation masks into LVIS-compatible 
Run-Length Encoding (RLE) and executes the official LVIS evaluation suite.

Author: Mahfuzzur Rahman
Course: COMP 6341 - Computer Vision (Concordia University)
Date: April 2026
"""

import numpy as np
import json
from lvis import LVIS, LVISEval
from pycocotools import mask as mask_utils

def mask_to_rle(binary_mask):
    """
    Converts a binary segmentation mask into LVIS-compatible RLE format.

    LVIS and COCO APIs require masks to be compressed into RLE strings for 
    efficient storage and IoU calculation. This function ensures the 
    underlying Fortran-contiguous memory layout required by pycocotools.

    Args:
        binary_mask (numpy.ndarray): A binary mask of shape (H, W).

    Returns:
        dict: A dictionary containing the 'size' and 'counts' of the RLE.
    """
    # pycocotools requires Fortran-order arrays for encoding
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    
    # Decode byte-string to utf-8 for JSON serializability
    rle['counts'] = rle['counts'].decode('utf-8') 
    return rle

def evaluate_lvis(gt_json_path, pred_json_path):
    """
    Performs standard LVIS segmentation evaluation against ground truth.

    This function utilizes the official LVIS API to calculate Average Precision (AP)
    across different IoU thresholds and object categories (Rare, Common, Frequent).

    Args:
        gt_json_path (str): Path to the LVIS ground truth annotations.json.
        pred_json_path (str): Path to the model's generated lvis_results.json.

    Returns:
        dict: The full results dictionary containing AP/AR metrics for 
              segmentation (segm).
    """
    print("📊 Loading LVIS Ground Truth...")
    lvis_gt = LVIS(gt_json_path)
    
    print("📥 Loading Predictions...")
    with open(pred_json_path, 'r') as f:
        # LVISEval accepts a list of annotation dicts directly
        preds = json.load(f)
    
    print("🧪 Running LVIS Evaluation (Segmentation)...")
    # Initialize the evaluator for segmentation (segm)
    lvis_eval = LVISEval(lvis_gt, preds, iou_type='segm')
    
    # Execute the matching and metric calculation pipeline
    lvis_eval.run()
    
    # Print the formatted summary table to terminal for immediate review
    print("\n" + "="*45)
    print("           LVIS EVALUATION SUMMARY")
    print("="*45)
    lvis_eval.print_results()
    
    # Return results for further downstream processing 
    return lvis_eval.results