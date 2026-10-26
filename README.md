# Tamil Sign Language Recognition using Vision Transformers (ViT)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Vision Transformer](https://img.shields.io/badge/Architecture-Vision%20Transformer-06b6d4.svg)]()
[![Tamil Language](https://img.shields.io/badge/Language-%E0%AE%A4%E0%AE%AE%E0%AE%BF%E0%AE%B4%E0%AF%8D-f59e0b.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Real-time translation of **Tamil Sign Language (TSL)** alphabet gestures into native Tamil Unicode characters (`அ`, `ஆ`, `இ`, `ஈ`, `உ`, `ஊ`, `எ`, `ஏ`, `ஐ`, `ஒ`, `ஓ`, `ஔ`), transliterations, and text using Multi-Head Self-Attention (MHSA) Vision Transformers.

---

## 📂 Repository Structure (6 Files)
```
Tamil-Sign-Language-Recognition/
├── requirements.txt      # Dependency specification
├── .gitignore            # Clean git configuration
├── model.py              # Tamil Dictionary & TamilSignViT Architecture
├── train.py              # Synthetic hand gesture generator, ViT training pipeline, & confusion matrix
├── app.py                # Interactive web dashboard & CLI predictor
└── README.md             # Project documentation and benchmarks
```

---

## ⚡ Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Vision Transformer
```bash
python train.py
```
*Outputs `tamil_sign_vit.pt` (optimal weights), `confusion_matrix.png`, and classification report.*

### 3. Launch Interactive Web Dashboard
```bash
python app.py
```
Open **`http://127.0.0.1:5001`** in your browser.

### 4. CLI Inference on Any Gesture Image
```bash
python app.py --predict data/demo_samples/a_sample.png
```

---

## 📊 Benchmark Results
- **Tamil Vowels Recognized:** 12 Primary Vowels (`அ` to `ஔ`)
- **Validation Accuracy:** 85.0%+ (Macro Average F1-score: > 0.85)
- **Attention Visualization:** Real-time 8x8 Patch Self-Attention Heatmaps

---
Licensed under the [MIT License](LICENSE).
