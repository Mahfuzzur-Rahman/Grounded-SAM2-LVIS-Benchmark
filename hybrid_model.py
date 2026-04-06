"""
Hybrid Grounded-SAM 2 Model Module
----------------------------------
This module implements a zero-shot object detection and segmentation pipeline 
by creating a 'handshake' between Grounding DINO and SAM 2.

Author: Mahfuzzur Rahman
Course: COMP 6341 - Computer Vision (Concordia University)
Date: April 2026
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# ==============================================================================
# 1. Environment & Path Configuration
# ==============================================================================
WORKSPACE_ROOT = "/workspace"
GSAM2_PATH = os.path.join(WORKSPACE_ROOT, "Grounded-SAM-2")
DINO_PACKAGE_PARENT = os.path.join(GSAM2_PATH, "grounding_dino")

# Inject local paths into sys.path to ensure module discoverability
if GSAM2_PATH not in sys.path:
    sys.path.append(GSAM2_PATH)
if DINO_PACKAGE_PARENT not in sys.path:
    sys.path.append(DINO_PACKAGE_PARENT)

# Component Imports
from groundingdino.util.inference import load_model, load_image, predict
import groundingdino.datasets.transforms as T
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ==============================================================================
# 2. Model Implementation
# ==============================================================================

class GroundedSAM2:
    """
    A unified wrapper for Open-Vocabulary Object Detection (Grounding DINO) 
    and High-Fidelity Segmentation (SAM 2).
    """

    def __init__(self):
        """
        Initializes the hybrid model by loading checkpoints and 
        moving weights to the appropriate compute device.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Define absolute paths for model weights and configurations
        self.sam2_checkpoint = "/workspace/Grounded-SAM-2/checkpoints/sam2_hiera_large.pt"
        self.sam2_model_config = "sam2_hiera_l.yaml"
        self.dino_checkpoint = "/workspace/Grounded-SAM-2/checkpoints/groundingdino_swint_ogc.pth"
        self.dino_config = "/workspace/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"

        # --- Initialize Segment Anything Model 2 (SAM 2) ---
        self.sam2_model = build_sam2(
            self.sam2_model_config, 
            self.sam2_checkpoint, 
            device=self.device
        )
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # --- Initialize Grounding DINO (Object Detector) ---
        self.dino_model = load_model(
            self.dino_config, 
            self.dino_checkpoint, 
            device=self.device
        )

    def run_inference(self, image_path, text_prompt, box_threshold=0.3):
        """
        Executes the detection-to-segmentation pipeline.

        Args:
            image_path (str): Local path to the input image.
            text_prompt (str): Natural language description of objects to find.
            box_threshold (float): Confidence threshold for Grounding DINO.

        Returns:
            tuple: (boxes_pixel, masks, phrases, logits)
        """
        # Data Sanitization: Limit prompt length to prevent token overflow
        words = text_prompt.split(". ")
        safe_prompt = ". ".join(words[:20])
        
        # Load and transform image for the detector
        image_source, image = load_image(image_path)

        # Stage 1: Zero-Shot Object Detection (Grounding DINO)
        boxes, logits, phrases = predict(
            model=self.dino_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=0.25
        )

        # Early exit if no objects are detected
        if len(boxes) == 0:
            return [], [], [], []

        # Stage 2: High-Fidelity Segmentation (SAM 2)
        self.sam2_predictor.set_image(image_source)
        
        # Coordinate Transformation: Convert normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
        h, w, _ = image_source.shape
        boxes_pixel = boxes * torch.Tensor([w, h, w, h])
        
        # Convert Center-XY-Width-Height to Top-Left/Bottom-Right
        boxes_pixel[:, :2] -= boxes_pixel[:, 2:] / 2  # Convert to x1, y1
        boxes_pixel[:, 2:] += boxes_pixel[:, :2]      # Convert to x2, y2
        
        # Generate segmentation masks using predicted boxes as spatial prompts
        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_pixel.numpy(),
            multimask_output=False
        )

        return boxes_pixel, masks, phrases, logits