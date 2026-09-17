# Lab 02: Answers

Haider Mushtaq (FA23-BAI-044)

**1. Which three pretrained models performed best in Lab Activity 1?**

From Lab 01 Table 1, ranked by test accuracy on the ISIC 9-class test set:

| Rank | Model | Accuracy | Macro-F1 | AUC |
|---|---|---|---|---|
| 1 | ResNet50 | 61.86% | 60.90% | 91.39% |
| 2 | DenseNet121 | 59.32% | 60.09% | 90.83% |
| 3 | ResNet101 | 58.47% | 57.22% | 92.24% |

ResNet50 led on accuracy and F1. DenseNet121 was a close second with the highest precision of all eight models and a fraction of the parameters (7.0M vs 23.5M). ResNet101 had the best AUC but lower accuracy, which suggests it ranks classes well but its argmax decisions were less sharp under the same 5-epoch budget. These three are used for every experiment in Lab 02.

**2. How does filtering affect each of the three models?**

Use one paragraph per model. State the baseline macro-F1, then the best and worst filter with the size of the change in percentage points. `summary.txt` prints exactly this.

**3. Which filter produces the greatest change compared with the unfiltered baseline?**

Take the filter with the largest mean |Δ macro-F1| across the three models from `delta_vs_baseline.csv`. Name it, give the number, and say whether the change was up or down.

**4. Does the effect of a filter remain consistent across all three models?**

Read the Δ macro-F1 heatmap. For each filter, say whether the sign is the same in all three columns. Where it flips, name the model that behaves differently and suggest why (depth, receptive field, presence of squeeze-excitation, etc.).

**5. Does filtering improve or decrease macro-F1 and balanced accuracy?**

Report the mean Δ for both metrics over all 15 filtered runs. Note whether any single filter improved both on any single model.

**6. Which lesion classes are most affected by filtering?**

Top three rows of `class_sensitivity.csv`. Relate them to what the class looks like: e.g. classes distinguished by fine texture or network patterns are expected to suffer most under smoothing.

---

**7. Why might smoothing remove useful lesion texture or morphological information?**

Smoothing is low-pass filtering. Every pixel becomes a weighted average of its neighbours, so high-frequency content, which is exactly where fine texture lives, is attenuated or removed. Dermoscopic diagnosis leans on high-frequency cues: the pigment network in a nevus, dots and globules, streaks at the border of a melanoma, the milia-like cysts of a seborrheic keratosis, arborising vessels in a basal cell carcinoma. These are a few pixels wide at 224 × 224. A 5 × 5 average filter has a support wider than most of them, so it blurs them into the surrounding skin tone. Border sharpness is also a diagnostic feature (an abrupt border is a melanoma warning sign), and smoothing turns an abrupt border into a gradual one.

The three smoothing filters degrade differently. The average filter treats all neighbours equally and also smears edges. The Gaussian filter weights the centre more, so it preserves slightly more structure at the same kernel size. The median filter is non-linear: it removes impulsive noise and hair artefacts while keeping step edges, so it is usually the least destructive of the three, but it still flattens texture.

**8. Why might sharpening or edge detection help or hurt classification?**

Sharpening adds a scaled high-pass version of the image back to itself. It amplifies borders, vessels and texture, which could help a model that under-weights those cues. It also amplifies noise, JPEG artefacts, hair and ruler markings, and it changes the pixel-intensity distribution the pretrained network was calibrated on. With a mild kernel the net effect is often close to zero; with a strong kernel it usually hurts.

Sobel is more drastic. It keeps only gradient magnitude and discards colour and absolute intensity. Colour is one of the strongest signals in dermoscopy: blue-white veil, red vascular lesions, brown versus black pigment. Throwing it away removes information the model relies on. The edge map also looks nothing like a natural photograph, so the ImageNet-pretrained early layers, which are tuned to colour and texture statistics of natural images, are now operating out of distribution. Some accuracy can be recovered during fine-tuning, but the expectation is a clear drop. Edge detection can help in narrow cases where shape or border irregularity alone is the discriminating feature, but for a 7-class dermoscopy problem it removes more than it adds.

**9. What is the difference between convolution and correlation?**

Both slide a kernel over an image and compute a weighted sum at each position. The difference is the orientation of the kernel.

Cross-correlation: `(I ⋆ K)(x, y) = Σ_i Σ_j I(x+i, y+j) · K(i, j)`

Convolution: `(I ∗ K)(x, y) = Σ_i Σ_j I(x−i, y−j) · K(i, j)`

Convolution flips the kernel both horizontally and vertically before applying it. For a symmetric kernel (average, Gaussian, the sharpening kernel used here) the two operations give identical results. For an asymmetric kernel like Sobel they differ by a sign flip in the output.

Convolution is the mathematically "proper" operation: it is commutative and associative, which is what makes the convolution theorem and the frequency-domain view of filtering work. `cv2.filter2D` actually computes correlation, which is why the OpenCV documentation says to flip the kernel yourself if you need true convolution. Deep learning frameworks also compute correlation in their "convolution" layers. It does not matter there because the kernels are learned, so the network simply learns the flipped version.

**10. Based on your results, explain the relationship between classical image processing and deep-learning-based feature extraction.**

A classical filter is a fixed, hand-designed convolution kernel. The first convolutional layer of a CNN is a bank of learned kernels of the same kind. Visualising the first layer of any ImageNet-trained network shows Gaussian-like blobs, oriented edge detectors that look like Sobel or Gabor filters, and colour-opponent filters. The network rediscovers classical image processing on its own, then stacks further layers to combine those primitives into texture, shape and part detectors.

That framing explains the experimental pattern. Applying a fixed filter before the network is equivalent to forcing an extra, non-learnable layer in front of it. If that layer removes information (smoothing, Sobel) the network cannot get it back, so performance drops. If it only re-weights information the network already extracts (mild sharpening) the network can compensate during fine-tuning and the change is small. The learned pipeline is strictly more flexible than any single hand-picked kernel, which is why the unfiltered baseline is expected to win or tie in most rows.

Where classical preprocessing still earns its place is in removing things the network should not learn: hair, vignetting, colour casts from different dermatoscopes, scale differences. Those are normalisation problems, not feature-extraction problems. The lesson from this lab is that preprocessing should target artefacts, not features, and that any filter which changes the appearance of the lesion itself should be treated as a risk rather than a default.
