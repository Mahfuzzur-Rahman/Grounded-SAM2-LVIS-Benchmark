# Grounded-SAM 2: Zero-Shot Open-Vocabulary Segmentation on LVIS

This repository contains a high-performance computer vision pipeline that integrates **Grounding DINO** and **Segment Anything Model 2 (SAM 2)** to perform zero-shot object detection and segmentation. This project specifically benchmarks performance on the **LVIS 16k dataset**, focusing on the model's ability to identify "Rare" categories without prior training.

---





## 📊 Benchmarking Results

The systems performance on the LVIS 16k subset demonstrates a significant advantage in zero-shot scenarios. Notably, the model achieved a higher accuracy on rare categories than common ones, validating the strength of the open-vocabulary approach.

| Metric | Result | Context |
| :--- | :--- | :--- |
| **mAP (Overall)** | **18.3%** | Robust base performance (IoU=0.50:0.95) |
| **mAP (Rare)** | **18.6%** | **Core Success: High zero-shot recall** |
| **mAP (Common)** | 16.9% | Baseline comparison for seen categories |
| **mAP (Frequent)** | 19.7% | Performance on high-density data |
| **Geometric Accuracy** | **19.4%** | AP75 Precision (High-fidelity masking) |
| **Avg Latency** | **173.7 ms** | Near real-time throughput on RTX 4090 |

---

## 🎓 Academic Context

* **Institution:** Concordia University
* **Course:** COMP 6341 - Computer Vision
* **Author:** Mahfuzzur Rahman
* **Student ID:** 40293992
* **Date:** April 2026

---
*Developed as part of the Master of Applied Computer Science (MACS) program at Concordia University.*
