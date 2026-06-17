# Breast Cancer & Cyst Classification Using ResNet50

This repository contains an end-to-end medical deep learning project designed to detect and classify breast ultrasound images into 3 distinct diagnostic categories: Normal, Benign, and Malignant.

## 🚀 Key Features
- **Advanced Image Preprocessing:** Integrates a Median Filtering pipeline to eliminate characteristic ultrasound speckle noise before feature extraction.
- **Data Imbalance Solution:** Solves class-imbalance using Random Oversampling techniques, shifting train samples to 891 optimized inputs.
- **ResNet50 Architecture:** Implements a state-of-the-art deep residual network with custom Top-Layers, regularized using L2 penalties and Dropout (0.4).
- **Modern Koyu Tema GUI:** A fully functional Tkinter desktop application providing immediate evaluation on loaded files or real-time external camera setups.

---

## 📁 Repository Structure
- `dataset/`: Sourced images mapped hierarchically across normal, benign, and malignant folders.
- `models/`: Targets best-performing checkpoints (`busi_cancer_ft_resnet_final_balanced.h5`).
- `busi_cancer_detection.ipynb`: Notebook pipeline containing complete training steps and the unified GUI architecture.

---

## 🛠️ Technical Workflow
1. **Speckle Noise Filtering:** Images pass through an adaptive $3\times3$ Median Filter to smooth out high-frequency sensor noise.
2. **Transfer Learning Phases:** 
   - **Stage 1:** Core ResNet50 base remains frozen while training dense custom classification layers with an initial LR of $1\times10^{-4}$.
   - **Stage 2 (Fine-Tuning):** Unfreezes non-batch-norm structural parameters to align generic ImageNet features with specific ultrasound textures using a micro learning rate of $5\times10^{-7}$.
3. **Application Interaction:** The user interface dynamically shifts alert colors (Green for Normal, Orange for Benign, Red for Malignant Risk flags) depending on multi-class array argmax probabilities.

---

## 📊 Performance & Evaluation
- **Overall Test Accuracy:** Achieve a verified test performance metric of ~`71.79%` across unseen validation datasets.
