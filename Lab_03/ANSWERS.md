# Lab 03 Answers

Haider Mushtaq (FA23-BAI-044)

## Discussion

### 1. Noise sensitivity

Sobel was most sensitive to salt-and-pepper noise in the measured comparison: its noise-sensitivity score was 79.54%. For Gaussian noise, unfiltered Sobel had a score of 64.74%, while Canny with Gaussian preprocessing had 73.24%. The score measures dissimilarity from the clean edge map, so higher values mean greater disruption.

### 2. Effect of Gaussian and median filtering

For Sobel, Gaussian filtering reduced the Gaussian-noise sensitivity from 64.74% to 53.05%. Median filtering reduced the salt-and-pepper sensitivity from 79.54% to 36.07%. The filtered salt-and-pepper case also had a small nonzero edge-quality score (0.06%). Canny's noisy cases remained sensitive in the saved table (73.24% for Gaussian noise and 74.83% for salt-and-pepper noise after the listed preprocessing).

### 3. Canny thresholds

The 30–100, 3 × 3 configuration produced the densest edge map, with 2,775 mean edge pixels and 14.17% heuristic quality. Raising thresholds to 100–200 reduced the count to 123 pixels and gave 24.65% quality. The best measured configuration was Canny-4: thresholds 50–150 with a 5 × 5 kernel, 33.16% quality and 343 mean edge pixels.

### 4. Edge maps and classification

Edge-only inputs reduced accuracy relative to raw images for all three classical classifiers. SVM changed from 77.95% raw to 72.70% filtered and 33.33% edge accuracy. Random Forest scored 77.43%, 72.70%, and 32.81%; KNN scored 77.43%, 72.18%, and 30.97%. The edge map removes color, intensity, and texture information that can help distinguish lesion classes.

### 5. Information loss

An edge map mainly retains intensity transitions. It can remove lesion color, absolute brightness, smooth-region information, and internal pigment or texture patterns. These features may be diagnostically useful even when they do not form a clear boundary.

### 6. CNN-learned features

A CNN learns filters from the task data and can combine edges with color, texture, and broader patterns. Manually replacing the image with an edge map fixes the representation in advance and discards information the network cannot recover. Learned early filters can respond to edge-like patterns while later layers retain access to the original image cues.

### 7. Most useful representation

Raw images were the most useful of the three conditions in this run. For the SVM, raw accuracy was 77.95%, compared with 72.70% for Gaussian-filtered images and 33.33% for Canny-edge images. This conclusion applies to the shared HAM10000 split used in this notebook. Lab 01 and Lab 02 headline results use different protocols, so their absolute scores should not be ranked directly against these values.

## Viva answers

- **An edge** is a location where image intensity changes rapidly.
- **First-order detectors** use image gradients; **second-order detectors** use changes in the gradient, such as the Laplacian.
- **Sobel x and y** measure intensity changes along perpendicular image directions.
- **The Laplacian is noise-sensitive** because taking second derivatives amplifies high-frequency variations.
- **Gaussian smoothing** suppresses high-frequency noise before edge detection.
- **Canny's advantage** is its multi-stage process: smoothing, gradient estimation, non-maximum suppression, and hysteresis thresholding.
- **Canny thresholds** define strong edges and weaker edges retained when connected to strong ones.
- **Gaussian noise** is additive continuous variation; **salt-and-pepper noise** changes sparse pixels to extreme dark or bright values.
- **Median filtering helps with salt-and-pepper noise** because isolated extremes are removed by the neighborhood median.
- **Edge detection can reduce classification performance** when it removes color, intensity, and texture cues.
- **CNNs can learn edge features** in their early convolutional layers.
- **Raw images may perform better** because the model can learn which color, texture, and shape cues matter.
