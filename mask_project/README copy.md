# Spot the Mask: Face Mask Image Classification

A deep learning solution for the [Zindi Spot the Mask Challenge](https://zindi.world/competitions/spot-the-mask) (#masks4all). The goal is to predict the probability that an image contains a person wearing a face mask. Submissions are scored with **AUC** (Area Under the ROC Curve).

**Author:** Witness Lubisi

## Results

| Item | Value |
|---|---|
| Validation AUC (20% stratified hold-out, 262 images) | **0.9991** |
| Zindi leaderboard | Rank 69 at the time of submission |

The validation set is small, so the validation AUC is a noisy estimate. The Zindi leaderboard score is the more reliable measure. This challenge has no public/private split.

## Approach

1. **Data check:** 1,308 labelled training images (664 no mask, 644 mask, almost perfectly balanced) and 509 test images. Every filename in the CSVs was matched against the image folder.
2. **Split:** 80/20 train/validation split, stratified on the target so both sets keep the same class ratio.
3. **Model:** transfer learning with a **ResNet50** pretrained on ImageNet (torchvision weights, openly available). The final layer is replaced with a single-logit output and trained with `BCEWithLogitsLoss`.
4. **Training in two phases:**
   - Phase 1: freeze the backbone and train only the new head (3 epochs, AdamW, lr 1e-3).
   - Phase 2: unfreeze everything and fine-tune (6 epochs, AdamW, lr 1e-4, cosine annealing). The checkpoint with the best validation AUC is kept.
5. **Augmentation (training only):** resize to 224x224, random horizontal flip, random rotation (10 degrees), colour jitter (brightness and contrast), ImageNet normalisation.
6. **Inference:** test-time augmentation (average of the original and horizontally flipped prediction). Output is a probability between 0 and 1 for each image, not a hard 0/1 label, because AUC depends on the ranking.

## Project structure

```
.
├── spot_the_mask.ipynb      # Full pipeline: data checks, training, prediction
├── requirements.txt         # Python dependencies
├── README.md
├── .gitignore               # Keeps competition data and model weights out of Git
├── images.zip               # NOT included (download from Zindi)
├── train_labels.csv         # NOT included (download from Zindi)
├── SampleSubmission.csv     # NOT included (download from Zindi)
└── data/images/             # Created when images.zip is extracted (NOT included)
```

Rename `spot_the_mask.ipynb` to match your actual notebook file name.

## How to reproduce

1. Create a free Zindi account, join the competition, and download `images.zip`, `train_labels.csv` and `SampleSubmission.csv` from the [Data page](https://zindi.world/competitions/spot-the-mask/data).
2. Open the notebook in Google Colab and switch to a GPU runtime (Runtime, Change runtime type, T4 GPU).
3. Upload the three data files to the Colab session (or mount a private Google Drive folder).
4. Install dependencies if needed (Colab already includes most of them):
   ```bash
   pip install -r requirements.txt
   ```
5. Run the notebook top to bottom. It extracts the images to `data/images/`, trains the model, saves `best_model.pt`, and writes `submission.csv`.
6. Upload `submission.csv` to Zindi.

Training takes a few minutes on a T4 GPU. A CPU-only run will be very slow.

## Competition rules compliance

This project was checked against the competition rules on the Zindi challenge page:

- **Only provided data:** only the competition images and labels are used. No external datasets.
- **Open-source packages only:** all dependencies are public open-source libraries (see `requirements.txt`). No AutoML tools are used.
- **Pretrained model:** the ResNet50 ImageNet weights are openly available to everyone through torchvision.
- **No metadata used:** image size, aspect ratio, pixel count and filenames are not used as model inputs. Every image is resized to a fixed 224x224 before it reaches the model. (Image sizes were only inspected during exploration, never as a feature.)
- **Competition data is not redistributed:** the images and CSV files are the property of Zindi and the competition host. They are not stored in this repository and must not be uploaded to GitHub, Kaggle or any other public site. Download them from Zindi yourself.
- **One account, own work:** submissions come from a single Zindi account.
- **Code sharing:** the Zindi rules say code must not be shared privately outside a team, and any shared code must be made available to all participants through the platform. If you share this repository, post the link on the competition discussion board so it is available to everyone.

Please read the [Terms of Use](https://zindi.world/terms) and the competition rules on the Zindi page, as they can be updated at any time.

## Possible improvements

- 5-fold cross-validation and averaging the fold models, so every labelled image is used for training.
- Ensembling different architectures (for example EfficientNet or ConvNeXt with ResNet50).
- Higher input resolution (288 or 320 pixels) so small masks are easier to see.
- Reviewing the most confidently wrong validation images to find label noise or hard cases.
- Richer test-time augmentation (multiple crops or scales).

## Acknowledgements

- [Zindi](https://zindi.world) and the competition host for the dataset and challenge.
- PyTorch and torchvision for the pretrained ResNet50 weights.
