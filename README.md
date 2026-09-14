Keyword Recognition using DS-CNN

A lightweight keyword spotting system that recognizes 8 spoken
commands from short audio recordings using a Depthwise Separable
Convolutional Neural Network (DS-CNN).

The project converts raw speech audio into spectrogram representations
and uses a compact CNN architecture designed for efficient audio
classification.

Project Overview

Keyword spotting is the task of detecting specific spoken commands from
short audio recordings.

This project recognizes the following 8 commands:

down

left

no

off

on

right

up

yes

The model is trained using the Google Speech Commands v0.02 dataset.

Pipeline

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

Model Architecture

The project uses a Depthwise Separable CNN (DS-CNN) architecture.

The model separates spatial filtering from channel mixing using
depthwise convolution followed by a 1×1 pointwise convolution. This
provides a lightweight architecture suitable for keyword spotting and
potential edge deployment.

Architecture

Input Spectrogram
       ↓
Resize → 32 × 32
       ↓
Normalization
       ↓
Conv2D
64 filters, 3×3
       ↓
DepthwiseConv2D
3×3
       ↓
Batch Normalization
       ↓
ReLU
       ↓
Conv2D
128 filters, 1×1
       ↓
Batch Normalization
       ↓
ReLU
       ↓
DepthwiseConv2D
3×3
       ↓
Batch Normalization
       ↓
ReLU
       ↓
Conv2D
256 filters, 1×1
       ↓
Batch Normalization
       ↓
ReLU
       ↓
Global Average Pooling
       ↓
Dropout (0.25)
       ↓
Dense
       ↓
8 Classes

Results

The trained model achieved approximately 93% overall accuracy on the
evaluation data.

Per-Class Performance

Keyword     Correct Predictions   Recall

down                350 / 383    91.4%
left                349 / 380    91.8%
no                  385 / 406    94.8%
off                 318 / 362    87.8%
on                  355 / 388    91.5%
right               359 / 382    94.0%
up                  333 / 356    93.5%
yes                 420 / 429    97.9%

The strongest class is yes, while the most noticeable confusion
occurs between off and up.

Training

The model was trained for 30 epochs.

The training curves show that:

Training accuracy increases rapidly during the initial epochs.

Validation accuracy stabilizes around 92--93%.

Training and validation accuracy converge closely.

Training and validation loss decrease substantially during training.

There is no significant indication of severe overfitting in the
later epochs.

The validation loss shows some fluctuations during the early epochs
before stabilizing as training progresses.

Dataset

This project uses the Speech Commands v0.02 dataset, containing
short spoken-word audio recordings across multiple command categories.

Only the following 8 classes are used:

down
left
no
off
on
right
up
yes

The complete dataset is approximately 2.3 GB, so it is not
included in this repository.

The notebook contains the logic required to download and extract the
dataset.

Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

SciPy

Jupyter Notebook

WSL / Ubuntu

Project Structure

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

Installation

Clone the repository:

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd keyword-recognition

Create a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Dataset Setup

The Speech Commands dataset should be downloaded separately.

The project expects the extracted dataset at:

data/speech_commands/

The dataset itself should not be committed to GitHub.

Running the Project

Launch Jupyter:

jupyter notebook

Open:

notebooks/simple_audio_8commands.ipynb

Run the notebook cells sequentially.

The notebook covers:

Environment setup

Dataset download and extraction

Audio loading

Audio preprocessing

Spectrogram generation

Dataset preparation

DS-CNN construction

Model training

Training and validation evaluation

Confusion matrix generation

Confusion Matrix

The confusion matrix shows strong diagonal dominance, indicating that
most samples are correctly classified.

The main source of confusion is:

off ↔ up

This suggests that these commands could benefit from additional data
augmentation, feature engineering, or architecture/hyperparameter tuning
in a future iteration.

Why DS-CNN?

A conventional convolution performs spatial filtering and channel mixing
together. A depthwise separable convolution decomposes this into:

Depthwise Convolution
        +
Pointwise (1×1) Convolution

This reduces computational cost while maintaining strong feature
extraction capability.

This makes DS-CNN particularly attractive for keyword spotting systems
intended for:

Embedded devices

Mobile applications

Edge AI devices

Voice-controlled systems

Resource-constrained hardware

Future Improvements

Possible extensions include:

Add more Speech Commands classes

Compare DS-CNN with a standard CNN

Experiment with MFCC features

Improve audio augmentation

Tune learning rate and other hyperparameters

Optimize the model using TensorFlow Lite

Apply post-training quantization

Measure inference latency and model size

Deploy the model for real-time microphone-based keyword detection

Test deployment on an embedded or edge device

Key Learning Outcomes

This project provided hands-on experience with:

Audio classification

Speech spectrograms

Audio preprocessing

Convolutional neural networks

Depthwise separable convolutions

Batch normalization

Global average pooling

Dropout

Model training and validation

Confusion matrix analysis

Keyword spotting

Lightweight neural network design

Practical ML project organization

Project Status

Completed

The current implementation successfully recognizes 8 spoken commands
with approximately 93% evaluation accuracy using a lightweight
DS-CNN architecture.

Author

Deepak Skandh

A hands-on project exploring Deep Learning, Audio Processing, Keyword
Spotting, and Edge AI.