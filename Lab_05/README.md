# Lab 05: HOG-Based Industrial Defect Detection

Haider Mushtaq (FA23-BAI-044)

This lab builds a classical computer-vision inspection pipeline for NEU-DET steel surface images. It extracts Histogram of Oriented Gradients (HOG) descriptors from grayscale images and compares an RBF support-vector machine with a random forest for binary defect detection and six-class defect classification.

## Contents

- `Lab05_HOG_Defect_Detection.ipynb`: complete experiment with saved outputs.
- `PAPER.md`: research-style report with the saved dataset, parameter-sweep, binary, and multiclass results.
- `requirements.txt`: packages used by this lab.
- `results/`: result tables and figures extracted from the notebook's saved outputs.

## Dataset and execution

The notebook downloads the public [NEU steel surface defect dataset](https://www.kaggle.com/datasets/sovitrath/neu-steel-surface-defect-detect-trainvalid-split) with `kagglehub`. It is intended to run in Google Colab. Open the notebook and run its cells in order; no Kaggle API token is required by the notebook.

The saved notebook uses 1,440 training images and 360 validation images, with 300 images per defect class. For binary classification, it creates 100 × 100 crops around annotated defects and up to two non-overlapping normal crops per source image. HOG descriptors are computed from grayscale images resized to 128 × 128.

## Results status

The saved run reports binary and six-class classifier results, and those values are included in `PAPER.md` and `results/`. The robustness and product-inspection cells do not have saved output in the committed notebook, so their measurements are not reported.
