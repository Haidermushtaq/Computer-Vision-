# Lab 02: Does image filtering help or hurt skin lesion classification?

Haider Mushtaq, FA23-BAI-044

In Lab 01 I benchmarked eight pretrained CNNs on the ISIC dataset. ResNet50, DenseNet121 and ResNet101 came out on top. This lab takes those three and asks a simple question: if I run classical filters over the images before training, does the model get better or worse?

I tested five filters (average, Gaussian, median, sharpening, Sobel) against an unfiltered baseline. Three models times six conditions is 18 training runs. Everything else stays fixed so the filter is the only thing that changes.

## What's in here

```
Lab_02/
├── Lab02_Filtering_HAM10000.ipynb    the whole pipeline
├── README.md
├── ANSWERS.md                        my answers to the lab questions
└── results/                          generated when the notebook runs
    ├── comparison_table.csv          the 18-row table the lab asks for
    ├── delta_vs_baseline.csv         how much each filter moved the numbers
    ├── class_sensitivity.csv         which classes react most to filtering
    ├── class_distribution.csv
    ├── reports/per_class.csv
    ├── curves/                       train/val history for every run
    ├── confusion/                    confusion matrix for every run
    └── figures/                      all the plots
```

## Dataset

HAM10000. It has 10,015 dermoscopy images in 7 classes, and the imbalance is brutal: nv alone is over 6,700 images while df has 115.

I didn't use the full thing. 18 runs on 10k images won't finish inside a Kaggle session, so I capped every class at 500. The three small classes (akiec, df, vasc) stay whole. That gives around 2,600 images. The exact per-class and per-split counts get printed in section 2 and saved to `results/class_distribution.csv`.

One thing I had to be careful about: HAM10000 has multiple photos of the same lesion under different `lesion_id`s. If you split randomly, near-identical images end up on both sides of the train/test line and your test accuracy is a lie. I split by lesion so that never happens. Roughly 72/14/14 train/val/test, stratified by class.

## How to run it

I ran this on Google Colab with a T4. Kaggle is only used as the download source for the dataset.

1. Runtime, Change runtime type, pick T4 GPU
2. Click the key icon in the left sidebar and add a secret called `KAGGLE_API_TOKEN` with your Kaggle API token (Kaggle, Settings, API, Create New Token). Turn on notebook access for it
3. Run `!pip install -q thop` once
4. Run all

The notebook reads the token from Colab secrets, pulls HAM10000 with `kagglehub`, and writes everything to `/content/lab02`. The token never touches the notebook file, so it's safe to commit.

Results go into `results.csv` after every single run. If Colab disconnects halfway, reconnect and rerun. It skips whatever already finished. Do copy `results.csv` to Drive between sessions though, because `/content` is wiped when the runtime dies.

It also runs on Kaggle if you attach `skin-cancer-mnist-ham10000` as an input. The notebook checks for that first and skips the download.

Budget about 3 to 4 hours on a T4 for all 18 runs. Free Colab may not give you that in one sitting, which is exactly why the checkpointing is there.

## Setup I kept fixed

| | |
|---|---|
| Input size | 224 x 224 |
| Where the filter sits | after resize, before augmentation |
| Augmentation (train only) | flips, rotation up to 20 degrees, mild colour jitter |
| Loss | cross-entropy with class weights and 0.1 label smoothing |
| Optimiser | AdamW, lr 1e-4, weight decay 1e-4, cosine decay |
| Epochs | 10 |
| Batch size | 32 |
| Which checkpoint | best validation accuracy |
| Metrics | accuracy, macro precision and recall, weighted F1, macro F1, balanced accuracy, macro AUC |

Filtering happens after the resize on purpose. A 5x5 kernel on a 600px image and a 5x5 kernel on a 224px image are not the same operation. Resizing first means every image sees the same amount of blur.

## The filters

| Filter | How |
|---|---|
| Average | `cv2.blur`, 5x5 |
| Gaussian | `cv2.GaussianBlur`, 5x5, sigma 1.5 |
| Median | `cv2.medianBlur`, size 5 |
| Sharpening | `cv2.filter2D` with the `[[0,-1,0],[-1,5,-1],[0,-1,0]]` kernel |
| Sobel | 3x3 Sobel in x and y, gradient magnitude, scaled to 0 to 255 |

Sobel gives a single grey channel. I copy it into three channels so the pretrained first conv layer still accepts it. Colour gets thrown away, and that's intentional. The Sobel condition is really asking "what happens when the model only gets edges."

## Things I'd change with more time

Ten epochs is on the short side. K-fold instead of a single split would make the comparisons more trustworthy. And the filter parameters (kernel size, sigma) were picked once and never tuned, so a gentler Gaussian might tell a different story.
