# Computer Vision

Coursework and portfolio projects in computer vision.

**Haider Mushtaq** (FA23-BAI-044)
BS Artificial Intelligence, COMSATS University Islamabad, Wah Campus

---

## Projects

### 01. Skin Cancer Classification (ISIC, 9 Classes)

A benchmarking study on automated skin lesion classification. The goal is to find out which pretrained CNN works best on a small, heavily imbalanced dermoscopy dataset, and whether classical machine learning models do better when they are given deep features instead of raw pixels.

**Three experiments:**

1. **Transfer learning comparison.** Eight ImageNet-pretrained CNNs fine-tuned on the dataset: AlexNet, VGG16, VGG19, ResNet18, ResNet50, ResNet101, DenseNet121, EfficientNet-B0.
2. **Deep features plus classical classifiers.** The best CNN is stripped of its final layer and used as a feature extractor. Seven classical models are then trained on those features: Logistic Regression, Decision Tree, Random Forest, KNN, Linear SVM, RBF-SVM, XGBoost.
3. **Computational efficiency.** Parameters, model size, FLOPs and single-image inference time for each architecture, so accuracy can be weighed against cost.

**Notebook:** [`notebooks/Task_01_Skin_Cancer_ISIC.ipynb`](notebooks/Task_01_Skin_Cancer_ISIC.ipynb)
**Methodology writeup:** [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)
**Dataset:** [Skin Cancer ISIC 9 Classes](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)

---

## Repository Structure

```
computer-vision/
├── notebooks/
│   └── Task_01_Skin_Cancer_ISIC.ipynb
├── docs/
│   └── METHODOLOGY.md
├── results/
│   ├── table1.csv
│   ├── table2.csv
│   ├── table3.csv
│   └── confusion_matrix.png
├── requirements.txt
└── README.md
```

---

## Stack

PyTorch, torchvision, scikit-learn, XGBoost, thop, pandas, matplotlib, seaborn

---

## Running the Code

The notebooks are written for Kaggle with a GPU accelerator.

1. Import the notebook into Kaggle
2. Add the dataset through **Add Input**
3. Set **Accelerator** to GPU and turn **Internet** on
4. Run all cells

The dataset path is detected automatically, so nothing needs to be edited before running.

For a local run:

```bash
pip install -r requirements.txt
```

Expect roughly 2 to 3 hours on a single P100 for the full eight-model training loop.
