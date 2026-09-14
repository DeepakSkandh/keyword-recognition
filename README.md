<div align="center">

# 🎙️ Keyword Recognition using DS-CNN

**A lightweight keyword spotting system that recognizes 8 spoken commands from short audio clips, powered by a Depthwise Separable CNN.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-93%25-blue)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

</div>

---

## 📑 Table of Contents

- [Overview](#-project-overview)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Training](#-training)
- [Dataset](#-dataset)
- [Technologies](#-technologies-used)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Running the Project](#-running-the-project)
- [Confusion Matrix](#-confusion-matrix)
- [Why DS-CNN?](#-why-ds-cnn)
- [Future Improvements](#-future-improvements)
- [Key Learning Outcomes](#-key-learning-outcomes)
- [Author](#-author)

---

## 🧭 Project Overview

**Keyword spotting** is the task of detecting specific spoken commands from short audio recordings. This project recognizes **8 commands** using the **Google Speech Commands v0.02** dataset:

<div align="center">

| `down` | `left` | `no` | `off` | `on` | `right` | `up` | `yes` |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|

</div>

<details>
<summary><b>🔀 View the processing pipeline</b></summary>

```text
Raw Audio
    ↓
Audio Preprocessing
    ↓
Spectrogram Generation
    ↓
Resizing & Normalization
    ↓
DS-CNN
    ↓
Classification
    ↓
8 Keyword Classes
```

</details>

---

## 🏗️ Model Architecture

The model uses a **Depthwise Separable CNN (DS-CNN)**, which separates spatial filtering from channel mixing via a depthwise convolution followed by a `1×1` pointwise convolution — a lightweight design suitable for keyword spotting and edge deployment.

<details open>
<summary><b>📐 View full architecture diagram</b></summary>

```text
Input Spectrogram
       ↓
Resize → 32 × 32
       ↓
Normalization
       ↓
Conv2D            64 filters, 3×3
       ↓
DepthwiseConv2D   3×3
       ↓
BatchNorm → ReLU
       ↓
Conv2D            128 filters, 1×1
       ↓
BatchNorm → ReLU
       ↓
DepthwiseConv2D   3×3
       ↓
BatchNorm → ReLU
       ↓
Conv2D            256 filters, 1×1
       ↓
BatchNorm → ReLU
       ↓
Global Average Pooling
       ↓
Dropout (0.25)
       ↓
Dense → 8 Classes
```

</details>

---

## 📊 Results

<div align="center">

### Overall Accuracy: **~93%**

</div>

<details open>
<summary><b>📈 Per-class performance (click to expand/collapse)</b></summary>

| Keyword | Correct Predictions | Recall |
|:---|:---:|:---:|
| `down`  | 350 / 383 | 91.4% |
| `left`  | 349 / 380 | 91.8% |
| `no`    | 385 / 406 | 94.8% |
| `off`   | 318 / 362 | 87.8% |
| `on`    | 355 / 388 | 91.5% |
| `right` | 359 / 382 | 94.0% |
| `up`    | 333 / 356 | 93.5% |
| **`yes`** | **420 / 429** | **97.9% 🏆** |

> ✅ Strongest class: **`yes`**
> ⚠️ Most noticeable confusion: **`off` ↔ `up`**

</details>

---

## 🏋️ Training

Trained for **30 epochs**.

<details>
<summary><b>📉 Training notes (click to expand)</b></summary>

- Training accuracy increases rapidly during the initial epochs.
- Validation accuracy stabilizes around **92–93%**.
- Training and validation accuracy converge closely.
- Training and validation loss decrease substantially during training.
- No significant indication of severe overfitting in later epochs.
- Validation loss fluctuates early on before stabilizing.

</details>

---

## 🗂️ Dataset

Uses the **Speech Commands v0.02** dataset, filtered down to 8 classes: `down`, `left`, `no`, `off`, `on`, `right`, `up`, `yes`.

> ⚠️ The complete dataset is **~2.3 GB** and is **not included** in this repository. The notebook contains the logic to download and extract it automatically.

---

## 🛠️ Technologies Used

<div align="center">

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/-Keras-D00000?logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C)
![SciPy](https://img.shields.io/badge/-SciPy-8CAAE6?logo=scipy&logoColor=white)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter&logoColor=white)
![WSL/Ubuntu](https://img.shields.io/badge/-WSL%2FUbuntu-E95420?logo=ubuntu&logoColor=white)

</div>

---

## 📁 Project Structure

```text
keyword-recognition/
│
├── data/
│   └── speech_commands/          # Dataset - not committed to Git
│
├── models/                       # Saved model files
│
├── notebooks/
│   └── simple_audio_8commands.ipynb
│
├── results/                      # Training plots and evaluation results
│
├── src/                          # Source code
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

<details open>
<summary><b>1️⃣ Clone the repository</b></summary>

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd keyword-recognition
```

</details>

<details>
<summary><b>2️⃣ Create a virtual environment</b></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
```

</details>

<details>
<summary><b>3️⃣ Install dependencies</b></summary>

```bash
pip install -r requirements.txt
```

</details>

<details>
<summary><b>4️⃣ Set up the dataset</b></summary>

The Speech Commands dataset should be downloaded separately and extracted to:

```text
data/speech_commands/
```

The dataset itself should **not** be committed to GitHub.

</details>

---

## ▶️ Running the Project

```bash
jupyter notebook
```

Then open:

```text
notebooks/simple_audio_8commands.ipynb
```

Run the notebook cells sequentially. The notebook walks through:

- [x] Environment setup
- [x] Dataset download and extraction
- [x] Audio loading
- [x] Audio preprocessing
- [x] Spectrogram generation
- [x] Dataset preparation
- [x] DS-CNN construction
- [x] Model training
- [x] Training and validation evaluation
- [x] Confusion matrix generation

---

## 🔎 Confusion Matrix

The confusion matrix shows strong diagonal dominance — most samples are correctly classified.

> **Main source of confusion:** `off` ↔ `up`

This suggests these commands could benefit from additional data augmentation, feature engineering, or architecture/hyperparameter tuning in a future iteration.

---

## ❓ Why DS-CNN?

A conventional convolution performs spatial filtering and channel mixing together. A **depthwise separable convolution** decomposes this into:

```text
Depthwise Convolution  +  Pointwise (1×1) Convolution
```

This reduces computational cost while maintaining strong feature extraction — making DS-CNN attractive for:

| Use Case | Fit |
|---|:---:|
| Embedded devices | ✅ |
| Mobile applications | ✅ |
| Edge AI devices | ✅ |
| Voice-controlled systems | ✅ |
| Resource-constrained hardware | ✅ |

---

## 🚀 Future Improvements

- [ ] Add more Speech Commands classes
- [ ] Compare DS-CNN with a standard CNN
- [ ] Experiment with MFCC features
- [ ] Improve audio augmentation
- [ ] Tune learning rate and other hyperparameters
- [ ] Optimize the model using TensorFlow Lite
- [ ] Apply post-training quantization
- [ ] Measure inference latency and model size
- [ ] Deploy the model for real-time microphone-based keyword detection
- [ ] Test deployment on an embedded or edge device

---

## 🎓 Key Learning Outcomes

<details>
<summary><b>Click to expand full list</b></summary>

- Audio classification
- Speech spectrograms
- Audio preprocessing
- Convolutional neural networks
- Depthwise separable convolutions
- Batch normalization
- Global average pooling
- Dropout
- Model training and validation
- Confusion matrix analysis
- Keyword spotting
- Lightweight neural network design
- Practical ML project organization

</details>

---

## ✅ Project Status

**Completed** — recognizes 8 spoken commands with ~93% evaluation accuracy using a lightweight DS-CNN architecture.

---

## 👤 Author

<div align="center">

**Deepak Skandh**

*A hands-on project exploring Deep Learning, Audio Processing, Keyword Spotting, and Edge AI.*

</div>