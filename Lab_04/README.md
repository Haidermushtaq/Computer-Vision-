# Lab 04: Skin Lesion Boundary Detection with Canny

Haider Mushtaq (FA23-BAI-044)

This lab applies grayscale conversion, Gaussian smoothing, Canny edge detection, and contour or morphology-based boundary extraction to skin-lesion images. It measures approximate lesion area and perimeter and compares filtering/edge-method combinations. The supplied brief is included as Lab_04_Assignment.pdf.

## Deliverables

- REPORT.md: report structure and experiment record.
- ANSWERS.md: prompts for the six assignment questions.
- results/boundary_measurements.csv: one measurement row for each of five images.
- results/method_comparison.csv: comparison of the eight required filtering/edge combinations.
- skin_lesion_canny.ipynb: existing code notebook on the repository main branch.
- Lab_04_Assignment.pdf: original assignment.

The result tables are blank templates. Complete them from actual images and measurements.

## Required workflow

Select five lesion images and preserve their filenames. Display each original, grayscale image, Gaussian-filtered image, three Canny threshold maps, and the selected boundary overlaid on the original. Explain the selected thresholds per image. Calculate area in pixels and perimeter in pixels using the same mask/contour convention for all images.

## Interpretation

These are pixel measurements, not physical dimensions. Without ground-truth masks, describe the output as an estimated boundary and avoid claiming segmentation accuracy. Record failures such as hair, ruler marks, low contrast, multiple contours, or incomplete lesion edges.
