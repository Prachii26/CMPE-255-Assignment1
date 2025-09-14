# 🌸Flower Classification (CRISP-DM, PyTorch, EfficientNet-B0)

A resource-efficient flower image classifier (daisy, dandelion, rose, sunflower, tulip) built end-to-end with the **CRISP-DM** methodology.  
Designed for **limited compute** (CPU-friendly), using **transfer learning** (EfficientNet-B0 / MobileNetV3-Small), **data augmentation**, and **regularization**.

**Validation (example)**: Acc **94.14%**, Macro-F1 **94.04%**, Top-3 **99.23%**.  


---

## 📌 Highlights
- **CRISP-DM**: Business Understanding → Data Understanding → Preparation → Modeling → Evaluation → Deployment  
- **Transfer learning**: EfficientNet-B0 (default) or MobileNetV3-Small (faster)  
- **Resource-aware**: small image size (224), modest batch, early stopping, mixed precision on CUDA  
- **Reproducible**: manifest & splits saved under `artifacts/`, config snapshot, deterministic loaders
- **Deployment-ready**: single/batch inference helpers, optional TTA, TorchScript/ONNX export

---

## 🧱 Repository Structure

├── FlowerClassificationModel.ipynb

├── artifacts/ # generated: configs, splits, metrics, plots, checkpoints

├── requirements.txt

├── .gitignore

├── README.me

---

## 🚀 Quickstart

### 1) Set up environment
```bash
# clone
git clone https://github.com/Prachii26/CMPE-255-Assignment1.git

# (recommended) create a virtual env
python -m venv .venv
source ./.venv/bin/activate        # Windows: .\.venv\Scripts\activate

# install deps
pip install -r requirements.txt
```
### 2) Open the notebook

Launch Jupyter/Lab and run the notebook top-to-bottom:

Chunk 1 – Setup & dataset download via KaggleHub

Chunk 2 – Data understanding, stratified splits, EDA

Chunk 3 – Data preparation: transforms, datasets, dataloaders

Chunk 4 – Modeling: EfficientNet-B0, freeze→finetune, early stopping

Chunk 5 – Evaluation: accuracy, macro-F1, confusion matrices, curves

Chunk 6 – Deployment: inference helpers, TorchScript/ONNX export

Dataset: Kaggle alxmamaev/flowers-recognition (downloaded automatically by the notebook via kagglehub).

---

## 📊 Results (example)

**Validation: Acc 94.14%, Macro-F1 94.04%, Weighted-F1 94.14%, Top-3 99.23%**

Check artifacts/classification_report_val.csv and artifacts/confusion_matrix_val_* for details.

___
## ⚙️ Configuration & Reproducibility

All tunables live in a Config dataclass (image size, batch size, LR, label smoothing, etc.).

A snapshot is saved to artifacts/config.json.

Stratified splits are exported to artifacts/train.csv, val.csv, test.csv.

---
## 🧪 Inference & Export

Use the Deployment chunk to:

Run single or batch predictions with top-k labels (optional TTA).

Export TorchScript (model_torchscript.pt) for portable CPU execution.

(Optional) Export ONNX for cross-runtime serving.

---

## 🏗️ Design Choices

**EfficientNet-B0**: best accuracy/FLOP trade-off; MobileNetV3-Small available for tighter CPU budgets.

**Augmentations**: RandomResizedCrop, flips, color jitter, light rotation.

**Regularization**: label smoothing, AdamW (weight decay), dropout, early stopping, gradient clipping.

**Scheduler**: OneCycleLR for fast, stable convergence in few epochs.

---

## 🙏 Acknowledgments

Dataset: alxmamaev/flowers-recognition

PyTorch & TorchVision teams for pretrained backbones.

---
## 📰 Blog Post
Medium: https://medium.com/@prachigupta2610/from-notebook-to-production-ready-a-crisp-dm-walkthrough-for-flower-classification-on-limited-79ca71de1018
