# Inferred RCNN (Pretrained Baseline)

This folder evaluates a **pretrained Mask R-CNN (ResNet-50 + FPN)** — without any fine-tuning — on the LVIS 16K subset. It serves as a closed-set baseline to contrast against the fine-tuned and zero-shot models.

---

## Files

| File | Description |
| :--- | :--- |
| `RCNN_Evaluation.ipynb` | Inference and evaluation notebook |
| `rcnn_results.json` | Raw per-image predictions (COCO format) |
| `final_rcnn_results.csv` | Final evaluation metrics |

---

## Notebook: RCNN_Evaluation.ipynb

### 1. Setup & Data Verification

- Installs dependencies and explores the `lvis_16k_dataset/` directory structure.
- Verifies image count and annotation file existence.
- Loads COCO-format annotations via `pycocotools`.

### 2. Model Loading

- Loads a **COCO-pretrained** Mask R-CNN (`pretrained=True`) directly — no custom head replacement or weight loading.
- Sets the model to evaluation mode on GPU.

### 3. Inference

- Defines a `get_predictions()` function that runs each image through the model.
- Filters predictions by a 0.5 confidence threshold and converts boxes to COCO `[x, y, w, h]` format.
- Iterates over all 16K images with a progress bar and saves results to `rcnn_results.json`.

### 4. COCO Evaluation

- Runs `COCOeval` (bbox IoU) to compute overall mAP and AP75.
- Splits categories into **rare**, **common**, and **frequent** groups (by annotation count terciles) and evaluates mAP per group.
- Patches missing `iscrowd` fields in annotations before evaluation.

### 5. Latency Measurement

- Measures per-image inference latency over 200 images and reports the average.

### 6. Results Export

- Aggregates all metrics into a DataFrame and saves to `final_rcnn_results.csv`.

---

## Evaluation Results

| Metric | Value | Context |
| :--- | :--- | :--- |
| Overall mAP | ~0.0 | IoU=0.50:0.95 |
| mAP (Rare) | 0.0 | Zero-Shot Success |
| mAP (Common) | 0.0 | Common Categories |
| mAP (Frequent) | 0.0 | Frequent Categories |
| mIoU Proxy (AP75) | ~0.0 | Geometric Accuracy |
| Avg Latency | 57.1 ms | 2x RTX 3090 |
| Baseline Type | Mask R-CNN | Closed-Set Comparison |

> **Key takeaway:** The pretrained Mask R-CNN (trained on 80 COCO classes) scores effectively **0.0 mAP** across all LVIS category groups. The COCO label space does not align with LVIS's 1,200+ categories, making the pretrained model unable to produce valid detections — demonstrating why fine-tuning or open-vocabulary approaches are necessary.

---

## Key Libraries

`torch`, `torchvision`, `pycocotools`, `PIL`, `numpy`, `pandas`, `tqdm`
