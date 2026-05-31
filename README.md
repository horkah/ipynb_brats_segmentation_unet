# Brain Tumor Segmentation with U-Net (BraTS 2020)

A 2D U-Net that segments brain tumors from multimodal MRI scans, trained on the
[BraTS 2020](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
dataset. The full workflow — data exploration, preprocessing, model, training,
and evaluation — lives in a single notebook:

**[`brain-tumor-segmentation-using-u-net.ipynb`](./brain-tumor-segmentation-using-u-net.ipynb)**

<!-- Optional: drop a prediction screenshot here -->
<!-- ![Prediction example](docs/prediction_example.png) -->

---

## Overview

Each patient in BraTS 2020 has four co-registered MRI modalities — **T1, T1ce, T2,
and FLAIR** — plus an expert segmentation mask. The model takes two of those
modalities (**FLAIR + T1ce**) as input channels and predicts a per-pixel class
for every axial slice.

| Label | Class             | Description                          |
|-------|-------------------|--------------------------------------|
| 0     | NOT tumor         | Healthy tissue / background          |
| 1     | Necrotic / Core   | Non-enhancing tumor core             |
| 2     | Edema             | Peritumoral edema                    |
| 3     | Enhancing         | Enhancing tumor (BraTS label 4 → 3)  |

## Pipeline

1. **Download & unzip** the dataset via the Kaggle CLI (training cases 000–100).
2. **Explore** the modalities and segmentation masks (slices, montages, per-class views).
3. **Preprocess** — min–max scale each volume to `[0, 1]`, resize slices to
   `128 × 128`, and keep `VOLUME_SLICES = 100` slices starting at index `22`.
4. **Split** the cases into train / validation / test (≈ 68% / 20% / 12%).
5. **Train** the U-Net with a Keras `Sequence` data generator that streams slices
   on the fly (low memory footprint).
6. **Evaluate** with Dice coefficient, Mean IoU, precision, sensitivity, and
   specificity, including per-class Dice scores.
7. **Predict & visualize** segmentations on held-out test patients.

## Model

A classic encoder–decoder **U-Net**:

- 4 down-sampling blocks (32 → 64 → 128 → 256 → 512 filters), each two `3×3`
  conv + ReLU layers followed by max pooling.
- Dropout at the bottleneck.
- 4 up-sampling blocks with skip connections back to the matching encoder stage.
- A final `1×1` conv with **softmax** over the 4 classes.

- **Input:** `(128, 128, 2)` — FLAIR and T1ce channels.
- **Loss:** categorical cross-entropy.
- **Optimizer:** Adam (`lr = 1e-3`), reduced on plateau.

## Getting Started

### Run on Kaggle (recommended)

This notebook was developed on Kaggle, where the dataset and GPU are available
out of the box. Add the
[BraTS 2020 dataset](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
to a GPU notebook and run all cells.

### Run locally

You'll need a GPU, the BraTS 2020 data, and a Kaggle API token (`kaggle.json`)
if you want the notebook to download the data for itself.

```bash
pip install tensorflow keras numpy pandas matplotlib opencv-python \
            nibabel scikit-image scikit-learn pydot kaggle
jupyter notebook brain-tumor-segmentation-using-u-net.ipynb
```

> **Note:** the dataset ships with a corrupted file name in
> `BraTS20_Training_355` — the notebook renames it automatically before loading.

## Key Parameters

| Parameter          | Value     | Meaning                                  |
|--------------------|-----------|------------------------------------------|
| `IMG_SIZE`         | `128`     | Slices are resized to `128 × 128`        |
| `VOLUME_SLICES`    | `100`     | Slices used per volume                   |
| `VOLUME_START_AT`  | `22`      | First slice index included               |
| `epochs`           | `25`      | Training epochs                          |
| Input channels     | FLAIR, T1ce | MRI modalities fed to the model        |

## Results

Training and validation curves (accuracy, loss, Dice, Mean IoU) are plotted from
the `CSVLogger` output, and the notebook visualizes predicted vs. ground-truth
masks for individual test patients across all tumor sub-regions. The best model
weights are checkpointed per epoch (the notebook loads the epoch-19 checkpoint
for inference).

## Repository Layout

```
.
├── brain-tumor-segmentation-using-u-net.ipynb   # the notebook
└── README.md
```

The notebook writes a few artifacts at runtime:

- `model_.{epoch}-{val_loss}.weights.h5` — per-epoch weight checkpoints
- `my_model.keras` — the full saved model
- `training.log` — per-epoch metrics (CSV)

## Notes on This Version

This is a cleaned-up version of the original Kaggle notebook:

- Centralized all file-path construction behind a single `imgPath()` helper.
- Removed dead and duplicated code (an unused/broken data loader and a redundant
  prediction function).
- Fixed cell-ordering issues so the notebook runs top-to-bottom.
- Reduced training to 25 epochs, which is sufficient on this small subset.

## Acknowledgements

- **Dataset:** [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html),
  mirrored on Kaggle by [awsaf49](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation).
- **Architecture:** U-Net (Ronneberger et al., 2015), *“U-Net: Convolutional
  Networks for Biomedical Image Segmentation.”*

## License

The code in this repository is released under the MIT License. The BraTS 2020
dataset is subject to its own terms — see the dataset page before redistributing.
