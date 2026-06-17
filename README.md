# Ultrasound Image Breast Cyst & Nodule Detection

This repository contains a deep learning-based medical image processing project designed to detect and classify cysts/nodules in breast ultrasound images. The project seamlessly integrates state-of-the-art computer vision models with a real-time graphical user interface (GUI) for deployment.

## 🚀 Key Features
- **Two-Stage Transfer Learning:** Features frozen pre-training and dynamic Fine-Tuning optimized for complex medical artifacts.
- **Data Augmentation Workflow:** Overcomes data imbalance using real-time geometric and brightness adjustments.
- **Interactive Tkinter GUI:** Allows non-technical users to load dynamic medical files or use live camera inputs to retrieve instant predictions.

---

## 📁 Repository Structure
- `data/`: Dataset folder categorized into `healthy` and `cystic` cases.
- `models/`: Destination directory for saved model weights (`.h5` checkpoints).
- `ultrasound_cyst_detection.ipynb`: Core monolithic notebook containing complete training loops and the GUI pipeline.

---

## 🛠️ Technical Deep Dive & Workflow

### 1. Data Preprocessing & Balancing
- Ultrasound images are normalized ($1/255$), automatically converted to RGB channels, and resized to $224 \times 224$ pixels to fit core model footprints.
- Mitigated class imbalance issues by computing balanced weights dynamically via `scikit-learn`.

### 2. Deep Learning Architecture
- **Base Network:** MobileNetV2 pre-trained on ImageNet weights for robust, lightweight feature extraction.
- **Top Layers:** Global Average Pooling 2D layer coupled with a Regularized Dense Layer (256 units, L2 penalty), Batch Normalization, and Dropout ($0.5$) to completely avoid overfitting.
- **Two-Stage Optimization:**
  - **Stage 1 (Feature Extraction):** Top layers trained with a frozen base model at a learning rate of $2\times10^{-5}$ over an initial phase.
  - **Stage 2 (Fine-Tuning):** Unfreezes the last 50 layers of MobileNetV2 to capture specific medical textures, optimized with a fine learning rate of $1\times10^{-6}$.

### 3. Desktop Application (GUI Deployment)
- Built using Python's Native **Tkinter** engine and optimized with `OpenCV` thread streams.
- Provides robust desktop interaction including file dialog image processing and live external camera predictions complete with percentage-based probability readouts.

---

## 💻 Tech Stack
- **Languages:** Python
- **Frameworks & Core Libraries:** TensorFlow (Keras), OpenCV, Pillow, NumPy, Scikit-Learn
- **UI Framework:** Tkinter

---

## 📊 Evaluation & Metrics
The model saves its absolute best weights during execution based on `val_accuracy` and maps performance through a clear classification matrix reporting Precision, Recall, and F1-Scores across `Sağlıklı` (Healthy) and `Kistli` (Cystic) classification labels.
