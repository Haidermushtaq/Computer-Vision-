# Lab 01: Skin Cancer Classification (ISIC, 9 Classes)

Haider Mushtaq (FA23-BAI-044)

Benchmarks eight pretrained CNNs on the ISIC 9-class dataset, then uses the best one as a frozen feature extractor for seven classical classifiers, then compares every architecture on parameters, size, FLOPs and inference time.

## Contents

```
Lab_01/
├── Task_01_Skin_Cancer_ISIC.ipynb
├── README.md
├── METHODOLOGY.md              # why each design choice was made
└── results/
    ├── table1.csv              # 8 CNNs: acc, macro P/R/F1, AUC
    ├── table2.csv              # ResNet50 features + 7 classifiers
    ├── table3.csv              # params, MB, GFLOPs, ms/image
    └── confusion_matrix.png    # ResNet50 on the test set
```

## Dataset

[Skin Cancer ISIC 9 Classes](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic), roughly 2,357 dermoscopy images across nine classes, shipped with Train and Test folders. A stratified 15% of Train is held out for validation. The Test folder is evaluated once, on the best-validation-epoch weights.

## Running

**Kaggle:** Add Input → `skin-cancer9-classesisic`. GPU on, Internet on. Run all.

**Colab:** T4 runtime. Add a `KAGGLE_API_TOKEN` secret (key icon, left sidebar). Run `!pip install -q thop` once. Run all.

The notebook finds the dataset on either platform without edits.

## Settings

Identical for all eight models: AdamW at 1e-4, weight decay 1e-4, cosine schedule, batch 32, mixed precision, class-weighted cross-entropy with label smoothing 0.1, flips/rotation/colour-jitter augmentation on train only. Model selection by validation accuracy.

The committed results were produced with `EPOCHS = 5` to fit the session budget. The notebook default is 20; longer training will raise the absolute numbers but is not expected to change the ranking.

## Results

**Table 1: transfer learning**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| AlexNet | 48.31 | 47.54 | 57.64 | 48.41 | 88.29 |
| VGG16 | 53.39 | 54.70 | 61.81 | 55.73 | 88.75 |
| VGG19 | 49.15 | 53.62 | 58.33 | 48.25 | 88.16 |
| ResNet18 | 56.78 | 54.17 | 58.56 | 54.20 | 89.96 |
| **ResNet50** | **61.86** | 60.47 | **65.74** | **60.90** | 91.39 |
| ResNet101 | 58.47 | 54.96 | 62.96 | 57.22 | **92.24** |
| DenseNet121 | 59.32 | **61.34** | 63.66 | 60.09 | 90.83 |
| EfficientNet-B0 | 52.54 | 51.18 | 55.09 | 50.38 | 88.43 |

**Table 2: ResNet50 features + classical classifiers**

| Classifier | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 60.17 | 65.08 | 58.33 | 56.52 | 91.38 |
| Decision Tree | 53.39 | 60.87 | 52.78 | 50.03 | 73.38 |
| Random Forest | 55.93 | 63.28 | 54.86 | 51.23 | 92.44 |
| KNN | 59.32 | 56.34 | 57.64 | 53.98 | 88.07 |
| **Linear SVM** | **61.86** | **65.48** | **59.72** | **57.66** | 91.85 |
| RBF-SVM | 59.32 | 62.74 | 57.64 | 55.30 | **93.04** |
| XGBoost | 56.78 | 61.32 | 55.56 | 53.13 | 89.48 |

**Table 3: computational efficiency**

| Model | Params (M) | Size (MB) | GFLOPs | ms/image | Accuracy |
|---|---|---|---|---|---|
| AlexNet | 57.04 | 217.59 | 1.42 | 1.88 | 48.31 |
| VGG16 | 134.30 | 512.30 | 30.93 | 8.93 | 53.39 |
| VGG19 | 139.61 | 532.56 | 39.26 | 10.83 | 49.15 |
| ResNet18 | 11.18 | 42.69 | 3.65 | 2.25 | 56.78 |
| ResNet50 | 23.53 | 89.95 | 8.26 | 5.77 | 61.86 |
| DenseNet121 | 6.96 | 26.88 | 5.79 | 20.57 | 59.32 |
| EfficientNet-B0 | 4.02 | 15.49 | 0.83 | 7.70 | 52.54 |

Inference measured on a Kaggle GPU, batch size 1, mean of 100 runs after 20 warm-up runs.

## Observations

- ResNet50 wins on accuracy and F1. DenseNet121 gets within 2.5 points using under a third of the parameters, which makes it the better choice if model size matters.
- Linear SVM on frozen ResNet50 features matches ResNet50's own fine-tuned head on accuracy and beats it on precision. The features carry most of the signal; the classifier on top is a secondary decision.
- DenseNet121's inference time is the outlier: fewest parameters and modest FLOPs but the slowest per-image latency, because its dense connectivity produces many small sequential operations that do not parallelise well on GPU. Parameter count and speed are not the same thing.
- VGG16/19 are the worst trade-off in every column: largest, slowest, and mid-table accuracy.
