# Lab 04 Report: Skin Lesion Boundary Detection with Canny

Haider Mushtaq (FA23-BAI-044)

## Abstract

This experiment tested a simple boundary-detection pipeline on five HAM10000 lesion images. Each image was converted to grayscale, smoothed with a 5 × 5 Gaussian filter, and processed with Canny at three threshold pairs. Morphological closing and contour scoring were used to select a candidate lesion boundary and estimate its area and perimeter. A valid contour was found for one image; the other four produced no valid contour under the implemented selection rule. For the successful image, the estimated area was 8,517 pixels and the perimeter was 1,154.1 pixels. The results show that the pipeline is not reliable enough for these images without improved preprocessing or a stronger boundary-selection method.

## 1. Introduction

The aim was to estimate a skin lesion's boundary from a dermoscopic image using grayscale conversion, Gaussian smoothing, Canny edge detection, and contour extraction. The experiment also compares several edge and filter combinations and estimates pixel area and perimeter.

The attached Skin Lesion Classification paper concerns classification accuracy and model efficiency, rather than boundary detection. Its numerical classification results are not used here. This report uses its formal report organization while reporting only the Lab 04 notebook's boundary-detection results.

## 2. Methodology

Five images were selected from HAM10000: three melanocytic nevus images and two melanoma images. The notebook resized each image, when needed, so its longest dimension was at most 600 pixels.

Each image was converted from BGR to grayscale and filtered with a 5 × 5 Gaussian kernel with sigma 1.5. Canny edge maps were generated at thresholds 50–100, 100–200, and 150–250.

For boundary extraction, the edge map was closed with a 7 × 7 elliptical structuring element for two iterations. External contours with area below 1% or above 80% of the image were discarded. Remaining contours were scored using area and solidity, with a penalty for contours touching the image border. If all candidate scores were zero, the notebook selected the threshold with fewer edge pixels as a tie-break. Area was calculated from the filled contour mask; perimeter was the closed contour arc length.

The comparison experiment evaluated Original, Average, Gaussian, and Median preprocessing followed by either Sobel or Canny. It scored noise handling, edge continuity, and contour solidity using image-derived proxies because no ground-truth lesion masks were available. The final labels Poor, Fair, and Good are relative rank groups among the eight tested combinations, not segmentation-accuracy measurements.

## 3. Experimental Setup

- Dataset: HAM10000
- Images: ISIC_0025438.jpg, ISIC_0029881.jpg, ISIC_0026399.jpg, ISIC_0028847.jpg, and ISIC_0027102.jpg
- Image selection: three nevus and two melanoma samples, selected with a fixed random seed
- Maximum image dimension: 600 pixels
- Gaussian filter: 5 × 5, sigma 1.5
- Canny thresholds: 50–100, 100–200, and 150–250
- Boundary method: morphological closing followed by external-contour scoring
- Measurements: filled-contour pixel count and closed-contour arc length

## 4. Results

### 4.1 Canny boundary measurements

| Image | Selected threshold | Area (pixels) | Perimeter (pixels) | Result |
|---|---:|---:|---:|---|
| Image 1 — ISIC_0025438.jpg | 100–200 | Not detected | Not detected | No valid contour |
| Image 2 — ISIC_0029881.jpg | 150–250 | Not detected | Not detected | No valid contour |
| Image 3 — ISIC_0026399.jpg | 50–100 | Not detected | Not detected | No valid contour |
| Image 4 — ISIC_0028847.jpg | 100–200 | Not detected | Not detected | No valid contour |
| Image 5 — ISIC_0027102.jpg | 50–100 | 8,517 | 1,154.1 | Valid contour |

For Images 1–4, every threshold configuration had a contour score of zero. Their displayed threshold selections were tie-break outcomes, not evidence that those thresholds detected a lesion boundary. The notebook stores area and perimeter as zero for these failures; the zeros are failure sentinels and must not be interpreted as actual lesion measurements. Image 5 had a score of 1,894 at 50–100 and zero at the other two threshold pairs, producing the only valid contour.

### 4.2 Filtering and edge-method comparison

| Method | Noise handling | Edge quality | Boundary detection | Overall performance |
|---|---|---|---|---|
| Original + Sobel | Fair | Fair | Fair | Fair |
| Original + Canny | Poor | Good | Fair | Poor |
| Average + Sobel | Good | Poor | Good | Good |
| Average + Canny | Poor | Poor | Poor | Poor |
| Gaussian + Sobel | Good | Fair | Good | Good |
| Gaussian + Canny | Poor | Poor | Poor | Poor |
| Median + Sobel | Good | Good | Good | Good |
| Median + Canny | Fair | Good | Poor | Fair |

Median + Sobel received the strongest proxy ratings across the three criteria. This does not establish that it produced accurate lesion masks: the ratings use edge-map and contour-shape proxies, and the experiment had no ground-truth masks. The direct Canny contour result remains the more important finding: four of the five images had no valid boundary.

## 5. Discussion

The three Canny threshold pairs did not produce valid contours for Images 1–4. Since all their contour scores were zero, the selected thresholds (100–200, 150–250, and 50–100, respectively) were determined by the edge-count tie-break. Image 5 showed a clearer response at 50–100, where the contour score was 1,894.

The pipeline therefore did not consistently separate the lesion from surrounding skin. Weak or incomplete boundaries, internal lesion structures, hair, and other image texture can create disconnected edges or competing contours. Morphological closing can join nearby edge fragments, but it cannot guarantee that the resulting outer contour is the lesion boundary. The comparison table's relative ratings also show that the filter/edge combination changes edge continuity and contour proxies; without expert masks, these ratings cannot tell whether the extracted outline is anatomically correct.

The four failed detections are a substantial limitation, not a minor measurement issue. Reporting zero as lesion area would be misleading, so the results table marks those measurements as not detected. A more reliable study should use expert segmentation masks, improve illumination and hair handling, test contour-selection rules, and compare the output against ground truth using overlap metrics such as Dice or intersection over union.

## 6. Conclusion

The implemented Gaussian-plus-Canny pipeline produced a measurable candidate boundary for one of five HAM10000 images. For that image, the estimated area was 8,517 pixels and the perimeter was 1,154.1 pixels. The other four images had no valid contour, so the present method did not meet the reliability needed for lesion-boundary measurement. Median + Sobel ranked best on the notebook's proxy comparison, but those proxy ratings are not a substitute for ground-truth segmentation evaluation.

## References

1. P. Tschandl, C. Rosendahl, and H. Kittler, “The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions,” Scientific Data, vol. 5, article 180161, 2018.
2. OpenCV documentation, image filtering, Canny edge detection, morphology, and contour operations.
3. The saved outputs in skin_lesion_canny.ipynb.
