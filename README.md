# Computer Vision

Lab work for the Computer Vision course, BS Artificial Intelligence.

**Haider Mushtaq** (FA23-BAI-044)
COMSATS University Islamabad, Wah Campus
Supervisor: Dr. Jamal Hussain shah



## Labs

| Lab | Topic | Dataset | Status |
|---|---|---|---|
| [Lab 01](Lab_01/) | Transfer-learning benchmark, deep-feature classifiers, computational efficiency | ISIC 9-class | Complete |
| [Lab 02](Lab_02/) | Effect of spatial filters on lesion classification | HAM10000 | Complete |
| [Lab 03](Lab_03/) | Edge detection, noise sensitivity, and classification | ISIC / HAM10000 | In progress |
| [Lab 04](Lab_04/) | Skin-lesion boundary detection using Canny | Skin lesion images | In progress |

---

## Lab 01: Skin Cancer Classification (ISIC, 9 classes)

Eight ImageNet-pretrained CNNs fine-tuned under identical settings, the best one used as a feature extractor for seven classical classifiers, and a cost/accuracy comparison across architectures.

**Top three by test accuracy**

| Model | Accuracy | F1 | AUC | Params |
|---|---|---|---|---|
| ResNet50 | 61.86% | 60.90% | 91.39% | 23.5M |
| DenseNet121 | 59.32% | 60.09% | 90.83% | 7.0M |
| ResNet101 | 58.47% | 57.22% | 92.24% | 42.5M |

Full tables: [`Lab_01/results/`](Lab_01/results/). Methodology: [`Lab_01/METHODOLOGY.md`](Lab_01/METHODOLOGY.md).

## Lab 02: Effect of Image Filtering on Skin-Lesion Classification (HAM10000)

Takes the three models above and retrains each under six conditions: unfiltered baseline plus average, Gaussian, median, sharpening and Sobel filters. Same lesion-grouped split, preprocessing and hyperparameters across all 18 runs.

**Headline result:** Sobel edge detection cost every model about 20 points of accuracy. The smoothing and sharpening filters moved Macro-F1 by less than 4 points in either direction, and which way depended on the model. The unfiltered baseline was the best or within 1.5 points of the best for all three. Filtering mostly hurt.

| Model | Best condition | Acc | Macro-F1 |
|---|---|---|---|
| ResNet50 | Gaussian | 76.90% | 77.90% |
| DenseNet121 | No filter | 74.80% | 75.89% |
| ResNet101 | Median | 76.90% | 77.61% |

Notebook, full 18-row table and figures: [`Lab_02/`](Lab_02/). Written answers: [`Lab_02/ANSWERS.md`](Lab_02/ANSWERS.md).

---

## Repository layout

```
Computer-Vision-/
├── README.md
├── requirements.txt
├── Lab_01/
│   ├── Task_01_Skin_Cancer_ISIC.ipynb
│   ├── README.md
│   ├── METHODOLOGY.md
│   └── results/
├── Lab_02/
│   ├── Lab02_Filtering_HAM10000.ipynb
│   ├── README.md
│   ├── ANSWERS.md
│   └── results/
├── Lab_03/
│   ├── Lab03_EdgeDetection_HAM10000.ipynb
│   ├── README.md
│   ├── REPORT.md
│   ├── ANSWERS.md
│   └── results/
└── Lab_04/
    ├── skin_lesion_canny.ipynb
    ├── README.md
    ├── REPORT.md
    ├── ANSWERS.md
    └── results/
```

---

## Environment

All notebooks run on Kaggle or Google Colab with a GPU. They detect the platform and locate the dataset automatically. On Colab, set a `KAGGLE_API_TOKEN` secret so the notebooks can download from Kaggle.

```
pip install -r requirements.txt
```

Stack: PyTorch, torchvision, OpenCV, scikit-learn, XGBoost, thop, pandas, matplotlib, seaborn.
