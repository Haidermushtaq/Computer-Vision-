# Methodology: Skin Cancer Classification (ISIC, 9 Classes)

This document explains what the notebook does and, more importantly, why each choice was made. Every design decision here has a reason behind it, and knowing the reason is what separates running code from doing research.

---

## 1. The Problem

Skin cancer is one of the most common cancers worldwide. Melanoma is the dangerous one. Caught early it is highly treatable, but manual diagnosis from dermoscopy images needs years of clinical experience and is still error prone. The idea behind automated classification is not to replace the dermatologist, it is to give them a second opinion that never gets tired.

The task here is a 9-way classification problem. Given a dermoscopy image, predict which of nine lesion types it shows.

---

## 2. The Dataset

The ISIC 9-class dataset from Kaggle contains roughly 2,357 images split into `Train` and `Test` folders, with nine class subfolders inside each.

Two properties of this dataset shape almost every decision in the notebook:

**It is small.** Around 2,000 training images is very little for deep learning. Training a CNN from scratch on this would overfit badly. This is the entire reason we use transfer learning.

**It is heavily imbalanced.** Some classes have several hundred images, others have well under a hundred. A model that ignores the rare classes entirely can still score decent overall accuracy, which makes raw accuracy a misleading metric here.

---

## 3. Data Splitting

The dataset ships with `Train` and `Test` folders but no validation set. The notebook carves a validation set out of `Train` using a stratified 85/15 split.

Why this matters:

- **Validation set** decides which epoch's weights to keep. Without it, you would have to either take the last epoch blindly or peek at the test set.
- **Test set is touched exactly once**, at the very end, on the best saved weights. If you validate on the test set, your reported numbers are optimistically biased and the whole study is methodologically unsound. This is the single most common mistake in student ML projects.
- **Stratified** means the class proportions are preserved in both halves. With rare classes present, a random split could easily put zero images of a rare class in validation.

---

## 4. Preprocessing and Augmentation

All images are resized to 224 by 224, which is what the ImageNet-pretrained models expect. Pixels are normalised using ImageNet mean and standard deviation, because the pretrained weights were learned on data normalised that way. Using different statistics would put the input distribution out of alignment with what the frozen early layers expect.

Augmentation is applied to the training set only:

| Transform | Why |
|---|---|
| Horizontal and vertical flip | Skin lesions have no natural orientation, so a flipped lesion is still the same lesion |
| Rotation up to 20 degrees | Same reasoning as flipping |
| Colour jitter | Handles differences in lighting and camera between clinics |
| Random affine shift and scale | Handles lesions that are off-centre or photographed at a different distance |

The validation and test sets get resize and normalise only. Augmenting evaluation data would make results non-reproducible, since you would score a different random version of each image every run.

---

## 5. Class Imbalance Handling

Two mechanisms are used together:

**Weighted loss.** Each class gets a weight inversely proportional to its frequency. Rare classes contribute more to the loss, so the model cannot simply ignore them. The formula is:

```
weight_for_class_i = total_samples / (num_classes * count_of_class_i)
```

**Macro averaging in the metrics.** Precision, recall and F1 are all computed with `average="macro"`. Macro averaging treats every class equally regardless of size. Weighted averaging, by contrast, would let good performance on the large classes hide terrible performance on the small ones. Macro is the honest choice for an imbalanced problem.

Label smoothing of 0.1 is also applied. It stops the model from becoming overconfident, which helps generalisation on small datasets.

---

## 6. Transfer Learning Setup

All eight models start from ImageNet weights. The only change is the final layer, which is replaced with a fresh `Linear` layer having nine outputs.

The whole network is then fine-tuned, not just the head. On a small dataset this is a judgement call. Freezing the backbone is faster and less prone to overfitting, but ImageNet features are photographs of everyday objects, which are quite different from dermoscopy images. Letting the whole network adapt usually wins here, and the augmentation plus weight decay keeps overfitting in check.

**Training settings, identical across all eight models:**

| Setting | Value | Reason |
|---|---|---|
| Optimizer | AdamW, lr 1e-4 | Low learning rate so pretrained weights are nudged, not destroyed |
| Weight decay | 1e-4 | Regularisation on a small dataset |
| Scheduler | Cosine annealing | Smoothly decays the learning rate to near zero by the last epoch |
| Epochs | 20 | Enough to converge without heavy overfitting |
| Batch size | 32 | Fits comfortably in P100 memory even for VGG19 |
| Mixed precision | Enabled | Roughly doubles training speed with no accuracy cost |

Keeping these identical matters. If VGG16 got a different learning rate to ResNet50, the comparison would be measuring hyperparameter tuning, not architecture.

The weights from the epoch with the best validation accuracy are saved, not the weights from the final epoch. A model can peak at epoch 12 and get slightly worse afterwards.

---

## 7. Experiment 1: Comparing Architectures (Table 1)

Eight architectures are trained and evaluated on the held-out test set. Five metrics are reported:

- **Accuracy** is the fraction of correct predictions. Simple, but inflated by the majority classes.
- **Precision** answers: of everything the model labelled as class X, how much really was class X. Low precision means false alarms.
- **Recall** answers: of all the real class X images, how many did the model find. Low recall on melanoma means missed cancers, which is the failure mode that actually matters clinically.
- **F1** is the harmonic mean of precision and recall. It only gets high when both are high.
- **AUC** measures how well the model separates classes across all decision thresholds, computed one-vs-rest. It is threshold independent, so it is more stable than accuracy on imbalanced data.

A confusion matrix is also produced for the best model. It shows exactly which classes get confused with which, which is far more informative than a single accuracy number. Confusions between visually similar classes are expected. Confusions where melanoma gets called benign are the ones to worry about.

---

## 8. Experiment 2: Deep Features Plus Classical Classifiers (Table 2)

A CNN is really two things stacked together: a feature extractor and a classifier. The convolutional layers turn an image into a vector of numbers describing what is in it. The final layer turns that vector into a class prediction.

This experiment cuts the CNN at that boundary. The final layer is replaced with `nn.Identity()`, so calling the model returns the feature vector instead of class scores. For ResNet50 that vector has 2048 numbers, for DenseNet121 it has 1024, and so on.

Those vectors then become the input to seven classical machine learning models. The question being asked is whether a different classifier on top of the same features beats the CNN's own trained final layer.

Two details worth understanding:

**No augmentation during feature extraction.** The training features are extracted using the evaluation transform. If you extract features from randomly augmented images, each image produces a slightly different vector each time, and the classical models end up learning noise.

**Scaling for distance-based and margin-based models.** Logistic Regression, KNN and both SVMs get standardised features. These algorithms are sensitive to feature scale, since they rely on distances or on regularisation penalties that assume comparable magnitudes. Tree-based models, Decision Tree, Random Forest and XGBoost, split on thresholds one feature at a time, so scaling makes no difference to them.

---

## 9. Experiment 3: Computational Efficiency (Table 3)

Accuracy alone does not tell you whether a model is deployable. A model meant to run on a clinic tablet or a phone has hard constraints on size and speed.

Four things are measured:

- **Parameters (millions).** The count of learnable weights. VGG16 has around 138 million, EfficientNet-B0 around 5 million, and they can score similarly. Bigger is not automatically better.
- **Model size (MB).** Parameters plus buffers, times 4 bytes for float32. This is roughly the download size for an app.
- **FLOPs (billions).** Floating point operations for one forward pass. This is a hardware-independent measure of compute cost. Note that `thop` reports MACs (multiply-accumulate operations), and one MAC is two FLOPs, which is why the notebook multiplies by 2.
- **Inference time (ms).** Wall-clock time for one image, averaged over 100 runs after 20 warmup runs.

The warmup runs are not optional. The first few GPU calls include kernel compilation and memory allocation overhead, and including them would inflate the timing badly. `torch.cuda.synchronize()` is also called before and after timing, because CUDA operations are asynchronous. Without it you would be timing how long it takes to queue the work, not to do it.

---

## 10. Known Limitations

Worth stating openly rather than leaving for someone else to point out:

- The dataset is small. Roughly 2,000 training images limits how much any of these conclusions generalise.
- ImageNet pretraining is a mismatch for medical imaging. Pretraining on a large dermoscopy corpus would almost certainly do better.
- A single train/test split means the numbers carry some variance. K-fold cross validation would give more trustworthy estimates but multiplies runtime by k.
- Hyperparameters were kept identical for fairness, not tuned per model. Each architecture might do better with settings chosen for it specifically.
- No external validation. Results on a different dataset from different clinics and different cameras would likely be lower.

---

## 11. References

- Tschandl, P., Rosendahl, C., Kittler, H. (2018). The HAM10000 dataset. *Scientific Data*.
- Codella, N. et al. (2018). Skin lesion analysis toward melanoma detection, ISIC challenge.
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Huang, G. et al. (2017). Densely Connected Convolutional Networks. *CVPR*.
- Tan, M., Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML*.
- Simonyan, K., Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR*.
