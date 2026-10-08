# Lab 03 Discussion and Viva

Haider Mushtaq (FA23-BAI-044)

Replace each bracketed prompt with observations from completed tables and figures. Do not state an empirical ranking without evidence from this run.

## Discussion questions

1. Which edge detector was most sensitive to noise? Use Table 1 to identify false or broken edges under each noise type.
2. How did Gaussian and median filtering affect edge quality? Compare continuity, sharpness, and false edges.
3. How did Canny thresholds change edge count and quality? Use Table 2 and identify the selected configuration.
4. Did edge-only images improve or reduce classification accuracy? Compare raw, filtered, and edge conditions on the same held-out split.
5. What information may edge maps lose? Consider color, absolute intensity, texture, and internal lesion patterns.
6. What advantages do CNN-learned early features have over manually supplied edge maps? Discuss task-dependent filters and preserved input information.
7. Which representation was most useful across Labs 01–03? Support the conclusion with comparable results. If datasets/splits differ, state that limitation.

## Viva quick review

- Edge: a rapid spatial change in image intensity.
- First-order methods use image gradients; second-order methods detect changes in gradients, often with zero crossings.
- Sobel x and y respond to vertical and horizontal intensity changes, respectively.
- The Laplacian is noise-sensitive because second derivatives amplify high-frequency variation.
- Gaussian smoothing reduces high-frequency noise before edge detection.
- Canny combines smoothing, gradient localization, non-maximum suppression, and hysteresis thresholding.
- The high Canny threshold selects strong edges; the low threshold retains weaker connected edges.
- Gaussian noise is continuous additive variation; salt-and-pepper noise is sparse extreme-valued pixels.
- Median filtering removes isolated salt-and-pepper pixels while often preserving step edges.
- Edge maps can hurt classification by discarding color, intensity, and texture cues.
- CNNs can learn oriented edge and texture filters in early convolutional layers.
- Raw images may perform better because they preserve cues the model can select for itself.
