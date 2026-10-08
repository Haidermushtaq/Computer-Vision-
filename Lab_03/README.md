# Lab 03: Edge Detection and Classification

Haider Mushtaq (FA23-BAI-044)

This lab compares classical edge detectors, studies how noise and smoothing affect their output, and evaluates raw, filtered, and edge-map inputs for classification. The supplied brief is included as Lab_03_Assignment.docx.

## Deliverables

- REPORT.md: report structure for methodology, experimental setup, results, discussion, conclusion, and references.
- ANSWERS.md: evidence prompts for the seven discussion questions and viva review.
- results/table1_noise_edge_quality.csv: noise and preprocessing observations.
- results/table2_canny_parameters.csv: threshold and kernel comparison.
- results/table3_cross_lab_classification.csv: cross-lab classifier comparison.
- Lab03_EdgeDetection_HAM10000.ipynb: existing code notebook on the repository main branch.
- Lab_03_Assignment.docx: original assignment brief.

The CSVs are blank result templates. Populate them from the experiment; no measurements are supplied here.

## Required experiment

Use representative images from at least three classes. Compare Sobel x/y/magnitude, Prewitt, Laplacian, LoG, and Canny. Add Gaussian and salt-and-pepper noise, compare no preprocessing with Gaussian and median smoothing, and evaluate at least three Canny threshold pairs. Then compare raw, Lab 02 filtered, and Lab 03 edge-map inputs with the same split, epochs, and metrics.

## Dataset and fair comparison

Lab 01 uses ISIC 9-class while Lab 02 uses HAM10000. The assignment asks for a common dataset and split across the comparison. State the selected dataset, class subset, split indices, and any mapping used to make the three conditions comparable. Do not combine scores from different test splits as if directly comparable.

## Suggested run order

1. Record the dataset, selected classes, and fixed train/validation/test split.
2. Generate the representative edge and noise figures.
3. Complete Tables 1 and 2 and select the edge configuration.
4. Train/evaluate raw, filtered, and edge conditions on the same split.
5. Complete Table 3, confusion matrices, comparison chart, report, and answers.
