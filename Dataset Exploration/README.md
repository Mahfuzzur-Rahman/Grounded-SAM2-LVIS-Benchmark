# Dataset Exploration

This folder contains three Jupyter notebooks used to explore, analyze, and prepare the LVIS dataset for benchmarking.

---

## Notebooks

### 1. LVIS_Dataset_Exploration.ipynb

Full exploration of the **LVIS v1 training set** (`lvis_v1_train.json`).

**What it does:**
- Loads the complete LVIS v1 annotations and prints high-level statistics (image count, annotation count, category count).
- Builds lookup maps for images, categories, and per-image annotations.
- Analyzes **class distribution** by frequency level (rare, common, frequent) and lists the top 20 categories.
- Visualizes random sample images with bounding box overlays and category labels.
- Computes detailed **annotation statistics**: objects per image (mean, median, std, min, max), bounding box dimensions, and area distributions.

**Key libraries:** `json`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `PIL`

---

### 2. Making Dataset Smaller.ipynb

Creates a **16k-image subset** of the full LVIS v1 training set with balanced category coverage.

**What it does:**
- Loads the full `lvis_v1_train.json` annotations.
- Performs **balanced sampling**: selects up to 5 images per category first, then fills the remainder globally to reach a target of 15,744 images.
- Copies images from local storage or **downloads missing images** from COCO URLs with retry logic.
- Handles duplicate detection and tracks copy/download/failure counts.
- Outputs the final subset as `annotations.json` with matched images and annotations in `lvis_16k_dataset/`.

**Key libraries:** `json`, `os`, `shutil`, `requests`, `tqdm`

---

### 3. Exploring Smaller Dataset.ipynb

Exploratory analysis of the **16k subset** produced by the previous notebook.

**What it does:**
- Loads `annotations.json` from the 16k subset and prints image, annotation, and category counts.
- Verifies data integrity by checking for images without annotations and annotations without images.
- Lists the **top 10 most frequent categories** by instance count and by number of images.
- Identifies categories with very few images (< 5) to assess coverage gaps.
- Plots **category frequency distribution**, **image size distribution** (width vs. height scatter), and **objects-per-image histogram**.
- Visualizes 5 random sample images with bounding boxes and category labels.

**Key libraries:** `json`, `os`, `cv2`, `matplotlib`, `collections`

---

## Data Paths

| Resource | Path |
| :--- | :--- |
| Full LVIS annotations | `lvis_v1_train.json` |
| 16k subset annotations | `lvis_16k_dataset/annotations.json` |
| 16k subset images | `lvis_16k_dataset/images/` |
| COCO source images | `train2017/` |
