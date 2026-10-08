# Lab 03 Report: Edge Detection and Its Effect on Classification

Haider Mushtaq (FA23-BAI-044)

## Abstract

This study compared five edge detectors on representative HAM10000 images, measured the effect of Gaussian and salt-and-pepper noise, tested Canny thresholds, and compared raw, Gaussian-filtered, and Canny-edge inputs for classification. Noise sensitivity was highest for Sobel on salt-and-pepper noise (79.54%). The best Canny configuration by the notebook's edge-quality heuristic was a 5 × 5 smoothing kernel with thresholds 50 and 150 (33.16%). On the shared 381-image test set, the SVM on ResNet50 features scored 77.95% accuracy on raw images, 72.70% on filtered images, and 33.33% on edge images. Edge-only input substantially reduced classification performance.

## 1. Introduction

Edges mark locations of rapid intensity change and can describe lesion borders and internal structures. This lab evaluates Sobel, Prewitt, Laplacian, Laplacian of Gaussian (LoG), and Canny detectors, then tests whether an edge-only representation retains the information needed to classify skin lesions.

## 2. Methodology

The notebook used HAM10000, containing 10,015 images across seven diagnostic classes. It capped each class at 500 images when needed, retaining smaller classes in full. The resulting subset contained 2,584 images. Images from the same lesion were kept together during a stratified group split: 1,813 train, 390 validation, and 381 test images.

For edge visualization, Sobel x, Sobel y, Sobel magnitude, Prewitt, Laplacian, LoG, and Canny were applied to representative images from at least three classes. Gaussian noise and salt-and-pepper noise were added to representative images. Gaussian and median filters were tested as denoisers. Noise sensitivity was calculated as 100 × (1 − Dice similarity) between the clean binary edge map and the corresponding noisy edge map. Edge quality was a heuristic based on edge density and connected-component continuity; it was not compared with expert edge annotations.

Canny was evaluated at three threshold pairs and with a larger Gaussian kernel. For classification, the same HAM10000 subset and split were used in all three conditions: raw RGB images, 5 × 5 Gaussian-filtered images with sigma 1.5, and Canny edge maps using the best configuration. The notebook trained ResNet50 and DenseNet121 and evaluated SVM, Random Forest, and KNN classifiers using ResNet50 deep features. CNN training used ten epochs on a Tesla T4 GPU.

## 3. Experimental Setup

- Dataset: HAM10000
- Subset: 2,584 images across akiec, bcc, bkl, df, mel, nv, and vasc
- Split: 1,813 train, 390 validation, 381 test; grouped by lesion
- Input size: 224 × 224 pixels for classification
- Gaussian noise: additive noise with standard deviation 25
- Salt-and-pepper noise: 3% of image pixels
- Gaussian and median denoising kernels: 5 × 5
- Canny configurations: 30–100 with 3 × 3, 50–150 with 3 × 3, 100–200 with 3 × 3, and 50–150 with 5 × 5
- Classification conditions: Raw, Filtered (Gaussian), and Edge (Canny)
- CNNs: ResNet50 and DenseNet121
- Classical classifiers: SVM, Random Forest, and KNN on ResNet50 features

## 4. Results

### 4.1 Noise and preprocessing

The following values are averages over one representative image per HAM10000 class. The edge-quality and noise-sensitivity columns are heuristic percentages from the notebook.

| Detector | Input | Noise | Preprocessing | Edge quality (%) | Noise sensitivity (%) |
|---|---|---|---|---:|---:|
| Sobel | Original | None | None | 0.00 | 0.00 |
| Sobel | Noisy | Gaussian | None | 0.00 | 64.74 |
| Sobel | Noisy | Salt and pepper | None | 0.00 | 79.54 |
| Sobel | Noisy | Gaussian | Gaussian filter | 0.00 | 53.05 |
| Sobel | Noisy | Salt and pepper | Median filter | 0.06 | 36.07 |
| Prewitt | Original | None | None | 0.00 | 0.00 |
| Laplacian | Original | None | None | 0.02 | 0.00 |
| LoG | Noisy | Gaussian | Gaussian filter | 0.00 | 62.18 |
| Canny | Original | None | Built-in smoothing | 14.17 | 0.00 |
| Canny | Noisy | Gaussian | Gaussian filter | 16.11 | 73.24 |
| Canny | Noisy | Salt and pepper | Median filter | 20.33 | 74.83 |

Sobel was most disrupted by salt-and-pepper noise (79.54%). Median filtering reduced its measured sensitivity to 36.07%, while Gaussian filtering reduced sensitivity to Gaussian noise from 64.74% to 53.05%. The Canny noisy-image rows remained substantially different from their clean references.

### 4.2 Canny parameter analysis

| Configuration | Low threshold | High threshold | Kernel | Edge quality (%) | Mean edge pixels |
|---|---:|---:|---:|---:|---:|
| Canny-1 | 30 | 100 | 3 × 3 | 14.17 | 2,775 |
| Canny-2 | 50 | 150 | 3 × 3 | 26.04 | 807 |
| Canny-3 | 100 | 200 | 3 × 3 | 24.65 | 123 |
| Canny-4 | 50 | 150 | 5 × 5 | 33.16 | 343 |

Canny-4 had the highest heuristic edge-quality score. The low thresholds produced a denser map; the high thresholds produced the sparsest map. The 5 × 5 smoothing configuration with thresholds 50–150 was selected for the classification edge condition.

### 4.3 Classification comparison

The classifier accuracies below come from the notebook's cross-condition table. Precision, recall, F1, training time, and inference time are reported for the Edge condition, as specified by that table.

| Model | Raw accuracy (%) | Filtered accuracy (%) | Edge accuracy (%) | Edge precision (%) | Edge recall (%) | Edge F1 (%) | Edge training (s) | Edge inference (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SVM | 77.95 | 72.70 | 33.33 | 27.92 | 27.77 | 30.31 | 22.32 | 3.91 |
| Random Forest | 77.43 | 72.70 | 32.81 | 27.47 | 28.35 | 29.77 | 15.28 | 0.23 |
| KNN | 77.43 | 72.18 | 30.97 | 28.43 | 27.16 | 29.85 | 0.00 | 0.44 |
| CNN Model 1 (ResNet50) | Not reported | Not reported | Not reported | Not reported | Not reported | Not reported | Not reported | Not reported |
| CNN Model 2 (DenseNet121) | Not reported | Not reported | Not reported | Not reported | Not reported | Not reported | Not reported | Not reported |

The saved cross-condition table contains blank/NaN CNN rows, so CNN scores are not presented as completed comparisons. The notebook's visible training log does include ResNet50 results for raw (77.95% accuracy, 78.00 F1), filtered (72.18%, 71.41 F1), and edge (31.23%, 30.58 F1), and DenseNet121 results for raw (73.75%, 73.67 F1) and filtered (75.59%, 75.25 F1). A complete DenseNet121 edge test result is not present in the saved output.

For the SVM, raw accuracy exceeded filtered accuracy by 5.25 percentage points and edge accuracy by 44.62 points. SVM had the highest mean accuracy across the three conditions (61.33%), but raw input was its strongest individual condition.

## 5. Discussion

Sobel was highly sensitive to salt-and-pepper noise, and median filtering reduced that disruption substantially. Gaussian filtering also reduced the measured effect of Gaussian noise on Sobel edges. These values describe similarity to the clean edge map, not clinical boundary correctness.

Increasing Canny thresholds reduced edge density. The 100–200 setting returned only 123 mean edge pixels, while 30–100 returned 2,775. The intermediate thresholds with 5 × 5 smoothing had the highest edge-quality heuristic score, balancing connected structure and edge density in this sample.

Raw images performed best in the classification comparison. Gaussian filtering reduced SVM accuracy modestly, while Canny-only images caused a large drop for all three classical classifiers. Edge maps preserve some outlines but discard color, absolute intensity, and texture. Those cues distinguish several HAM10000 classes, so the reduced input information is a plausible explanation for the lower scores.

The experiment compares all three conditions on one lesion-grouped split, but the Lab 01 and Lab 02 headline results use different datasets/protocols. The cross-condition comparison in this notebook is therefore the fair within-lab result; it should not be presented as a direct recalculation of the earlier labs' test scores.

## 6. Conclusion

The selected Canny setup was thresholds 50–150 with a 5 × 5 kernel according to the notebook's edge-quality heuristic. Median filtering reduced Sobel's sensitivity to salt-and-pepper noise, and Gaussian filtering reduced Sobel's sensitivity to Gaussian noise. For classification on the shared split, raw images were strongest and edge-only images were substantially worse. The edge-quality metric is heuristic, and the CNN rows in the saved cross-condition table are incomplete, so those limitations should be considered when interpreting the results.

## References

1. P. Tschandl, C. Rosendahl, and H. Kittler, “The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions,” Scientific Data, vol. 5, article 180161, 2018.
2. OpenCV documentation for filtering, Canny edge detection, and image morphology.
3. PyTorch and scikit-learn documentation for model training and classification.
