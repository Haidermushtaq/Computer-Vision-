# Lab 03 Report: Edge Detection and Classification

Haider Mushtaq (FA23-BAI-044)

## Introduction

State the objective of comparing edge detectors and testing whether edge maps retain useful information for skin-lesion classification.

## Methodology

Describe the dataset and classes, image selection, grayscale conversion, noise generation, Gaussian and median filtering, Sobel, Prewitt, Laplacian, LoG, and Canny implementations. Explain how Canny thresholds and the lesion-classification input were chosen.

## Experimental Setup

- Dataset and source:
- Selected classes and number of images:
- Train/validation/test split:
- Noise distributions and random seed:
- Filter parameters:
- Canny thresholds and kernel sizes:
- Classifiers/models and training settings:
- Evaluation metrics:
- Hardware/runtime:

Keep the split and training settings fixed across raw, filtered, and edge-map conditions.

## Results

Insert the completed tables from results/. Include the required edge-detector figure sequence for at least three classes, noise/preprocessing comparisons, confusion matrices for raw, filtered, and edge inputs, and a comparison chart.

## Discussion

Discuss edge continuity, sharpness, false/broken edges, noise sensitivity, smoothing effects, Canny thresholds, and classification performance. Answer the seven questions in ANSWERS.md using measured evidence.

## Conclusion

Summarize which edge detector and image representation were most useful, with the limits of this experiment.

## References

List dataset source, libraries, and any papers or documentation used.
