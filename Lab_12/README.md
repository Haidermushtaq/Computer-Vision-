# Lab 12: CNN Feature Visualization, Parameter Audit & Critical Analysis

## About the lab

This lab builds a small CNN pipeline by hand, without using a framework for the forward pass.
Five custom 3x3 kernels are written from scratch and applied through four stages: Conv, ReLU, MaxPool and AvgPool.
The results are checked against TensorFlow to confirm the hand written convolution and pooling are correct.
The second part audits the trainable parameters of two small CNNs and compares a 3x3 design against a 5x5 design.
The third part answers five critical thinking questions with experiments rather than opinion.
Tools: Python, NumPy, TensorFlow/Keras, Google Colab.

## How to run

Open `CNN_Feature_Visualization_Parameter_Audit.ipynb` in Google Colab, then choose Runtime > Run all.
No dataset downloads are needed. The notebook builds its own test image.
It takes about 2 minutes on CPU. No GPU is required.

## Part I summary

Five 3x3 kernels were implemented:

| Kernel | Sum | Has negative weights |
|---|---|---|
| Sobel-X (horizontal gradient) | 0.00 | Yes |
| Sobel-Y (vertical gradient) | 0.00 | Yes |
| Sharpen | 1.00 | Yes |
| Gaussian blur | 1.00 | No |
| Emboss | 1.00 | Yes |

Each kernel runs through four stages, giving a 5x4 grid of feature maps:

| Stage | Operation | Shape |
|---|---|---|
| A | Convolution | 256 x 256 |
| B | ReLU | 256 x 256 |
| C | MaxPool 2x2 | 128 x 128 |
| D | AvgPool 2x2 | 128 x 128 |

The main test image is a synthetic building facade with windows, mullions, ledges and a sloped roof.
A real photo is run through the same pipeline as a second check.
The hand written convolution matches `tf.nn.conv2d` for all five kernels, with maximum error below 1e-4.

Note on naming: Sobel-X measures horizontal change, so it detects vertical edges.
SciPy flips the kernel and returns the opposite sign. CNN frameworks do not flip, so this lab does not flip either.

## Part II: parameter audit

Formula used for every convolution layer: `(Kh x Kw x Cin + 1) x Cout`.

| Layer | Model A (3x3, MaxPool) | Model B (5x5, AvgPool) |
|---|---|---|
| Conv1 | 140 | 380 |
| Conv2 | 460 | 1260 |
| Pool | 0 | 0 |
| GAP | 0 | 0 |
| Dense | 110 | 110 |
| **Total** | **710** | **1750** |

Increase from A to B: `(1750 - 710) / 710 x 100 = 146.48%`.
Every manual count matches `model.summary()` exactly.
Padding and input size change the spatial shapes but not the parameter counts, because weights depend only on kernel size and channel counts.

## Answers to Q1 to Q5

### Q1. What does ReLU remove, and what are the alternatives?

* After ReLU the left sides of the windows go black in the Sobel-X map, because those edges give negative responses.
* In the Sobel-Y map the tops of the windows, the ledges and the sloped roof outline go black for the same reason.
* About 50% of the edge signal is lost. Sobel-X loses 50.0% of total absolute response and Sobel-Y loses 51.1%.
* Example pixel at row 70, column 32: the raw value is -2.611 and ReLU turns it into 0.
* The negative sign means a bright to dark edge. A hand check gives `4 x (0.12 - 0.78) = -2.64`, close to the measured -2.611.
* Leaky ReLU with slope 0.1 turns -2.611 into -0.261, so the edge stays faint and the sign is kept. Sigmoid maps flat wall areas to 0.5 and squeezes everything into (0, 1), which causes vanishing gradients.

### Q2. When is max pooling the wrong choice?

* Max pooling keeps thin bright lines. The one pixel window bar under the Sharpen kernel gives max 2.65 against avg 1.31 in the same 2x2 window.
* For bright features on a dark background, max pooling is the safer default.
* The picture reverses for dark ink on white paper, because the useful signal is the low value, not the high one.
* After two levels of pooling on a sign crop, max pooling leaves 0 pixels darker than 0.5.
* Average pooling leaves 36 and min pooling leaves 89, so the text survives.
* For dark on light, average pooling is the better choice, and min pooling is better still.

### Q3. Is a bigger kernel worth the parameters?

* Moving from 3x3 to 5x5 raises the total from 710 to 1750, an increase of 146.48%.
* Two stacked 3x3 layers reach the same 5x5 receptive field as one 5x5 layer. This was confirmed by a gradient test, not just by theory.
* With 10 channels throughout, one 5x5 layer costs 2510 parameters and two 3x3 layers cost 1820.
* That saves 690 parameters, or 27.5%, and adds a second non linearity.
* The saving is not automatic. Using a 10 channel middle layer inside Model B's Conv2 costs 1370 against 1260, so the stack is actually larger.
* The middle layer width decides whether stacking saves anything.

### Q4. Does the order of ReLU and pooling matter?

* Conv > ReLU > MaxPool gives the same result as Conv > MaxPool > ReLU, because ReLU is monotonic and does not change which value is largest.
* The patch `[-3, 2, -5, 1]` gives 2 in both orders.
* This was verified on all five real feature maps and on a random TensorFlow tensor. The outputs were identical every time.
* The equivalence does not hold for average pooling. The same patch gives 0.75 one way and 0 the other.
* Pooling first means ReLU runs on four times fewer values, dropping from 25,690,112 evaluations to 6,422,528, a saving of 75%.
* So the order is free to choose for max pooling, and pooling first is cheaper.

### Q5. Why do filters collapse, and how is it fixed?

* Cause 1 is constant initialization. Setting every weight to 0.05 makes all filters identical, with mean cosine similarity 1.000 between them.
* All filters then receive the same gradient and stay identical, so validation accuracy sits at 0.21.
* The fix is He initialization, which breaks the symmetry and reaches validation accuracy 1.000 with mean cosine similarity 0.331.
* Cause 2 is a learning rate that is too high. At lr = 0.5 all 5 ReLU filters die, 100% of activations are zero and accuracy falls to 0.11.
* The fix is to lower the learning rate to 0.001, or to keep lr = 0.5 and switch to Leaky ReLU.
* Leaky ReLU recovers validation accuracy 1.000 with 0 dead filters, because a dead unit still passes a small gradient.

## Conclusion

* The hand written convolution and pooling match TensorFlow, which confirms the pipeline is correct and not just plausible looking.
* ReLU throws away about half of the edge information from a zero sum kernel, so the choice of activation is a real design decision and not a default.
* Parameter count depends only on kernel size and channels, and two stacked 3x3 layers usually beat one 5x5 layer, though the middle layer width decides the saving.
* Filter collapse has clear causes that can be measured with cosine similarity and dead filter counts, so it can be diagnosed instead of guessed at.
