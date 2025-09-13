# Assignment 1 — Flower Classification (CRISP-DM, Transfer Learning)

A lean, reproducible image classification project on the Kaggle **Flowers Recognition** dataset using **TensorFlow/Keras**. It follows the **CRISP-DM** methodology end‑to‑end, is optimized for **limited compute**, and ships with **explainability (Grad-CAM)** and **deployment (SavedModel & TFLite)**.

---

## 📦 Quick Start

> You can run the whole pipeline from the provided notebook cells (Chunks 1–7). The project is modular; each chunk is self‑contained and documented.

```bash
# (Optional) Install deps in your environment
pip install -U kagglehub tensorflow pandas scikit-learn matplotlib pillow tqdm gradio

# Start Jupyter and open the notebook
jupyter notebook
```

**Data download (in-notebook):**
```python
import kagglehub
path = kagglehub.dataset_download("alxmamaev/flowers-recognition")
print("Kaggle dataset path:", path)
```

---

## 🗂️ Repository / Folder Structure

Suggested structure for your GitHub repo:

```
assignment-1/
├── README.md                           # ← this file
├── notebooks/
│   └── FlowerClassification.ipynb      # your working notebook
├── models/
│   ├── best_model.keras                # trained best model (from Chunk 3/6)
│   └── id2label.json                   # label maps (Chunk 3)
│       label2id.json
├── export/                             # deployment artifacts
│   ├── savedmodel/                     # TF SavedModel export (Chunk 6)
│   ├── model_fp32.tflite               # TFLite FP32 (Chunk 6/6b)
│   └── model_float16.tflite            # TFLite float16 (preferred on CPU)
├── scripts/
│   └── flower_cli.py                   # tiny CLI for inference (Chunk 7)
└── requirements.txt                    # optional pinning for reproducibility
```

> When you run the notebook, artifacts are created in `./models` and `./export`. Move them into your repo structure as shown above before committing.

---

## 🧭 CRISP-DM Walkthrough & How to Run

Each chunk corresponds to a CRISP‑DM phase and provides ready‑to‑run cells:

1) **Business & Data Understanding (Chunk 1)**  
   - Goal: accurate flower classification with low compute.  
   - Actions: environment setup, dataset download, quick EDA (class balance & samples).

2) **Data Preparation (Chunk 2)**  
   - Stratified **train/val/test** splits, efficient **tf.data** pipelines, **class weights**, and light **on‑model augmentation**.

3) **Modeling (Chunk 3)**  
   - **Transfer learning** with **MobileNetV2 (default)** or **EfficientNetB0**.  
   - Two‑phase training: **warm‑up (frozen backbone)** → **fine‑tune (top layers)**.  
   - Fixed LR handling (float LRs; `ReduceLROnPlateau` compatible).  
   - Artifacts saved: `models/best_model.keras`, `models/id2label.json`, `models/label2id.json`.

4) **Evaluation (Chunk 4)**  
   - Test **accuracy** & **Top‑3**, **classification report**, **confusion matrices**, and **misclassification gallery**.

5) **Explainability (Chunk 5)**  
   - **Grad‑CAM** utilities (final robust version) with gallery & single‑image helpers.  
   - Works reliably by **rebuilding the forward pass** for a single connected graph.

6) **Deployment — Exports (Chunk 6 & 6b)**  
   - Build a **fresh inference‑only model** (no augmentation) and **copy weights**.  
   - Export **SavedModel** and convert to **TFLite** (FP32 + float16) with robust fallbacks.  
   - Parity checks via cosine similarity and TFLite sanity run.

7) **Deployment — Interfaces (Chunk 7)**  
   - **Unified predictor** (Keras & TFLite).  
   - **CLI** script (`scripts/flower_cli.py`) and an optional **Gradio** mini‑app.

---

## 🧪 Reproducibility

- Global seed: `42` (see `set_global_seed`).  
- Recommended versions (print cell provided in Chunk 6):
  - Python ≥ 3.10
  - TensorFlow ≥ 2.15 / Keras 3+
  - numpy, pandas, scikit‑learn, matplotlib, pillow, tqdm, gradio
- Suggested `requirements.txt`:
```
tensorflow>=2.15
keras>=3.0.0
numpy
pandas
scikit-learn
matplotlib
pillow
tqdm
kagglehub
gradio
```

---

## 🧠 Model Choice & Training Process

- **Backbones:** MobileNetV2 (fast, ~3.5M params) or EfficientNetB0 (~5.3M).  
- **Why:** small yet strong on small datasets; ideal for CPU‑only training.  
- **Process:** warm‑up head with frozen backbone → unfreeze top ~20% (BatchNorm frozen) → fine‑tune with lower LR.  
- **Callbacks:** EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.  
- **Input:** 224×224 RGB, `[0,1]`, with in‑model augmentation.

---

## 📊 Evaluation Artifacts (examples)

- `classification_report.csv` (optional export from Chunk 4)  
- `confusion_matrix.png` and `confusion_matrix_normalized.png` (optional save)  
- Misclassification gallery figure(s)

> You can add a small save snippet around the plotting functions to persist figures into `./export/` for Git commits.

---

## 🔍 Explainability (Grad‑CAM)

- Robust Grad‑CAM implementation that computes gradients over the **backbone feature map** within a single graph.  
- **Gallery** for random test images and **single‑image** function for quick inspection.  
- Use to validate that the model attends to petals/disc florets rather than background.

---

## 🚀 Deployment

### SavedModel
```python
best_model.export("export/savedmodel")  # or best_model.save(..., save_format="tf")
```

### Build a fresh inference‑only model (no augmentation), copy weights
> See Chunk 6 cell “Build fresh inference‑only model”. This prevents conversion issues and matches training preprocessing.

### TFLite conversion (robust)
- FP32 and float16 models saved to `export/model_fp32.tflite` and `export/model_float16.tflite`.  
- The converter attempts: **builtins → SELECT_TF_OPS → concrete function** automatically.

### CLI (TFLite)
```bash
python scripts/flower_cli.py path/to/image.jpg --topk 3   --tflite export/model_float16.tflite
```

### Gradio App (optional)
```python
# in notebook
demo.launch(share=False)
```

---

## ⚠️ Troubleshooting (What we fixed)

- **LR schedule vs `ReduceLROnPlateau`:** created optimizer with **float LR** (not a schedule) so the callback can adjust it.  
- **Grad‑CAM `KeyError` / graph issues:** rebuilt forward pass inside the CAM function; ensured a single, connected graph; added a fallback gradient path.  
- **TFLite converter errors:** created a **fresh inference‑only** model (no augmentation), copied weights, and added **robust converter** fallbacks (SELECT_TF_OPS / concrete function).  
- **Input rank mismatch:** avoided reusing layer instances in a new graph; built a fresh model to prevent extra batch dims.

---

## ✅ Deliverables Checklist

- [ ] Notebook with Chunks 1–7 (`notebooks/FlowerClassification.ipynb`)  
- [ ] Trained model: `models/best_model.keras`  
- [ ] Label maps: `models/id2label.json`, `models/label2id.json`  
- [ ] SavedModel export: `export/savedmodel/`  
- [ ] TFLite models: `export/model_fp32.tflite`, `export/model_float16.tflite`  
- [ ] CLI: `scripts/flower_cli.py`  
- [ ] (Optional) Gradio UI ready to launch

---

## 📈 Results (fill with your run)

- Test Accuracy: `…`  
- Top‑3 Accuracy: `…`  
- Macro F1: `…`  

Include a confusion matrix and a few Grad‑CAM overlays demonstrating correct focus.

---

## 📚 Acknowledgments

- Dataset: Kaggle — **Flowers Recognition** by *alxmamaev*.  
- Backbones: MobileNetV2, EfficientNetB0 (TensorFlow/Keras Applications).

---

## 🔐 License

This repository is for coursework (**Assignment 1**). For the dataset, follow Kaggle’s terms of use.

---

### Appendix: Minimal End‑to‑End Script (outline)

If you later want to script this outside a notebook, the flow is:
1) Data indexing → splits → tf.data pipelines.  
2) Build model → warm‑up → fine‑tune → save artifacts.  
3) Evaluate (report + confusion matrices).  
4) Build inference‑only model → export SavedModel → convert to TFLite.  
5) Inference via CLI/Gradio.
