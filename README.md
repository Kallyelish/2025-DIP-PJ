# FSDerain: Frequency–Spatial Dual-Branch Fine-Tuning for Real-World Image Deraining

## 🔍 Overview

Single image deraining in real-world scenarios remains challenging due to the domain gap between synthetic and real rainy images. While models like **Restormer** perform well on synthetic datasets, they struggle to generalize to complex real-world scenes.

We propose **FSDerain**, a **Frequency–Spatial Dual-Branch Fine-Tuning** framework to improve Restormer's performance in real-scene deraining without modifying its backbone. Our method introduces two complementary components:

- 🌌 **Spatial-domain branch**: Preserves structure and semantics using L1, SSIM, and perceptual losses.
- ⚡ **Frequency-domain branch**: High-frequency rain patterns via FFT-based features and contrastive learning.

Extensive evaluations on both paired and unpaired real-world datasets show that our method enhances deraining quality, outperforming Restormer.

---
## 🗃️ Dataset Preparation

We support both **paired** and **unpaired** real-world datasets. Please follow the steps below:

### 🔹 Paired Dataset (e.g., `LHP-Rain`)

```bash
Dataset/
└── Real/
  ├── test/
  └── train/
```

### 🔹 Unpaired Dataset (e.g., `RE-RAIN`)

```bash
datasets/
└── RE-RAIN/
  ├── 1.png/
  ├── 2.png/
  └── ...

```

You can download these two datasets from [LHP-Rain](https://yunguo224.github.io/LHP-Rain.github.io/) and [RE-RAIN](https://pan.baidu.com/share/init?surl=0Y77VgT1Gs1D3hz5XJj8og&pwd=o9px).
You can also download the pre-trained **Restormer** model for fine-tuning from [here](https://drive.google.com/drive/folders/1ZEDDEVW0UgkpWi-N4Lj_JUoVChGXCu_u).

---

## 🏃 Training and Evaluation

### 🔧 Training

To fine-tune FSDerain using both spatial and frequency branches:

```bash
python new-dual.py
```

### 🚀 Inference

Inference on a real-world dataset:

```bash
python test.py --checkpoint ./checkpoints/derain_best.pth --save_dir ./results_real
```
### 🔍 Evaluation

Evaluation on a real-world dataset:

```bash
python metrics.py --pred ./results_real --gt [your_gt_path]
```
---

## 📌 Notes

- Our framework fine-tunes Restormer **without modifying its architecture**.
- Future directions include further modifying the network to address the issue of rain–background confusion caused by the original Restormer architecture.

---

## 💬 Contact

For questions or collaborations, please open an issue or contact:  
📧 22307130359@m.fudan.edu.cn
