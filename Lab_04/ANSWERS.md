# Lab 04 Answers

Haider Mushtaq (FA23-BAI-044)

### 1. Gaussian filtering before Canny

Gaussian filtering reduces high-frequency image noise before gradient-based edge detection. This can reduce false edges, but excessive smoothing can also weaken a faint lesion border.

### 2. Effect of the three Canny threshold settings

The notebook selected 100–200 for Image 1, 150–250 for Image 2, 50–100 for Images 3 and 5, and 100–200 for Image 4. For Images 1–4 all three candidate contour scores were zero, so the selected thresholds were tie-breaks based on edge count, not successful boundary detections. Image 5 had a contour score of 1,894 at 50–100 and zero at the other settings.

### 3. Best threshold

Canny 50–100 produced the only valid lesion contour, for Image 5. No tested threshold produced a valid contour for Images 1–4, so the experiment does not support one universally best threshold.

### 4. Use of edges for lesion detection

Edges can outline a lesion where its intensity or color transition against surrounding skin is clear. In this run, the method succeeded for only one of five images, showing that the boundary contrast and edge continuity were insufficient for the current contour-selection rule in most samples.

### 5. Observed problems

Four images had no valid lesion contour. Internal structures, weak or incomplete borders, hair, and other texture can produce fragmented edges or competing contours. The fifth image produced an estimated area of 8,517 pixels and perimeter of 1,154.1 pixels.

### 6. Possible improvements

Improve illumination and hair handling, test adaptive thresholding and morphology settings, use a contour-selection rule informed by lesion shape and location, and validate against expert masks. Ground-truth masks would allow meaningful Dice and intersection-over-union evaluation rather than proxy ratings.
