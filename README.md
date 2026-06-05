# 🍄 Mushroom Harvest Detection System

A computer vision pipeline for multi-class mushroom detection using YOLOv8, trained on merged public datasets from Roboflow Universe. Built as a foundation for automated agricultural monitoring systems.

---

## Project Goal

Detect and classify mushroom types in images using object detection. This is **Phase 1** of a two-phase system:
- **Phase 1 (this repo):** Detect mushrooms and classify by type
- **Phase 2 (upcoming):** Binary harvest-readiness classifier on top of detected boxes

---

## Model

- **Architecture:** YOLOv8s (Small) — transfer learning from ImageNet pretrained weights
- **Framework:** Ultralytics
- **Training:** Google Colab (Tesla T4 GPU)
- **Epochs:** 50 | **Image size:** 640 | **Batch:** 16

### Results

| Class | Precision | Recall | mAP50 |
|---|---|---|---|
| pink_oyster | 0.893 | 0.992 | 0.993 |
| oyster_mushroom | 0.908 | 0.957 | 0.962 |
| mushroom | 0.681 | 0.610 | 0.625 |
| **Overall** | **0.827** | **0.853** | **0.860** |

---

##  Dataset

7 public datasets merged from Roboflow Universe — 1,149 images total across 4 classes.

| Dataset | Source | Class |
|---|---|---|
| Pink Oyster Harvest Time v2 | Roboflow Universe | pink_oyster |
| Pink Oyster Harvest Time v1 | Roboflow Universe | pink_oyster |
| Oyster Mushroom 014 | Roboflow Universe | oyster_mushroom |
| Oyster Mushroom 04 | Roboflow Universe | oyster_mushroom |
| Oyster Mushroom 09 | Roboflow Universe | oyster_mushroom |
| Mushroom Detection (king bolete) | Roboflow Universe | king_bolete |
| Mushroom v2 | Roboflow Universe | mushroom |

**Split:** 961 train / 107 valid / 81 test

---

## Project Structure
## 🗂️ Project Structure

```
mushroom-harvest-detection/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_data.py       # Download all 7 datasets from Roboflow
│   │   └── merge_datasets.py      # Merge + remap class labels across datasets
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py               # YOLOv8s training with augmentation config
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── predict.py             # Run inference on test set or single image
│   │   └── evaluate.py            # Metrics, confusion matrix (in progress)
│   └── deployment/
│       └── __init__.py            # FastAPI endpoint (in progress)
├── configs/                       # Training hyperparameters (in progress)
├── demo/                          # Gradio web demo (in progress)
├── notebooks/                     # Analysis and visualizations (in progress)
├── models/                        # Trained weights — gitignored, on HuggingFace
├── datasets/                      # Downloaded data — gitignored
├── runs/                          # Training outputs — gitignored
├── requirements.txt
└── README.md
```

## Setup & Run

```bash
# Clone
git clone https://github.com/studtannta9166/mushroom-harvest-detection.git
cd mushroom-harvest-detection

# Install
pip install -r requirements.txt

# Download datasets (requires Roboflow API key)
python src/data/download_data.py

# Merge datasets
python src/data/merge_datasets.py

# Train
python src/training/train.py
```

---

## 🛠️Tech Stack

Python · YOLOv8 · PyTorch · Roboflow · OpenCV · Google Colab · Git

---

##  Roadmap

- [x] Multi-source dataset pipeline
- [x] YOLOv8 multiclass detection
- [ ] EfficientNet-B0 binary harvest classifier
- [ ] FastAPI inference endpoint
- [ ] HuggingFace Spaces live demo
- [ ] Real farm data collection and retraining

---

## License

Datasets used under CC BY 4.0. Model weights available on request.

## Planned Deployment

Camera-based real-time detection system. The model will be connected to a 
physical camera mounted above mushroom growing beds to automatically flag 
when mushrooms are visually ready for harvest.

Status: Model trained ; currently working on more data collection from real farm and imporving results ; Camera deployment: in August 2026
## Training Environment

- Platform: Google Colab
- GPU: Tesla T4 (15GB VRAM)
- Python: 3.12
- Framework: Ultralytics YOLOv8 8.4.60, PyTorch 2.11