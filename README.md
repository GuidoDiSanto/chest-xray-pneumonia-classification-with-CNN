# Chest X-Ray Pneumonia Classification

**Comparing custom CNNs and MobileNetV2 through group-aware data splitting, validation-based threshold selection, and error analysis.**

An exploratory deep learning project for binary classification of pediatric chest X-rays as **NORMAL** or **PNEUMONIA**. The notebook compares model architectures and training components, with particular attention to the trade-off between sensitivity and specificity.

**Author:** Guido Di Santo

[Explore the notebook](progetto.ipynb) · [Dependencies](requirements-progetto.txt) · [Recorded test results](artifacts/progetto/test_results.csv)

## Overview

- Compare a CNN with `Flatten`, a CNN with `GlobalAveragePooling2D`, and ImageNet-pretrained MobileNetV2.
- Audit filename-derived groups and exact pixel duplicates before splitting the data.
- Select decision thresholds on validation data using a target recall of at least 98%.
- Evaluate accuracy, sensitivity, specificity, precision, F1, balanced accuracy, ROC-AUC, and average precision.
- Estimate uncertainty using a group bootstrap and inspect false positives and false negatives.
- Run ablation experiments to investigate the effects of augmentation and class weights.

## Dataset and splitting

The project uses [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), downloaded through KaggleHub with **dataset version 2** pinned in the code.

The original `val` folder contains only 16 images. Validation data are therefore drawn from the original training folder using an approximately 80/20 stratified group split. The original test folder remains the test set.

The audit combines filename-derived identifiers with hashes of decoded grayscale pixels. Connected groups stay together, development images linked to test groups are excluded, and additional exact copies within the development set are removed. Filename-derived groups are a precaution: they do not certify patient identity, and the hash check does not rule out near-duplicates.

The saved run uses these partitions after the audit:

| Partition | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Training | 1,072 | 2,718 | 3,790 |
| Validation | 268 | 679 | 947 |
| Test | 234 | 390 | 624 |

The split can vary with library versions. The notebook saves the actual split manifests and environment metadata for each execution.

## Models and experimental protocol

| Model | Main feature | Total parameters |
|---|---|---:|
| CNN-Flatten | Three convolutional blocks followed by `Flatten` and a dense classifier | 8,024,193 |
| CNN-GAP | Same CNN with global average pooling replacing `Flatten` | 110,721 |
| MobileNetV2 | ImageNet pretraining, frozen-base training, then partial fine-tuning | 2,259,265 |

Two additional experiments remove augmentation or class weights from CNN-GAP, one component at a time.

Images are resized to **180 × 180**, preserving aspect ratio through padding. CNN inputs are scaled to `[0, 1]`; MobileNetV2 receives three replicated grayscale channels scaled to `[-1, 1]`. Training augmentation includes small rotations, zoom changes, and contrast adjustments.

1. **Training:** learn model weights with early stopping and learning-rate reduction.
2. **Validation:** select each model's threshold to minimize false positive rate while achieving recall ≥ 0.98. Select the primary model by validation specificity at that threshold, then ROC-AUC, then fewer parameters.
3. **Testing:** evaluate the fixed models and thresholds. Also report the standard 0.5 threshold as a reference.

The target recall is an experimental constraint, not a clinically validated requirement or a guarantee of test performance. Class weights are computed from training labels only.

## Recorded results

The table below uses each model's **validation-selected threshold**. These are results from the saved run, not averages across multiple seeds. The majority-class baseline predicts PNEUMONIA for every image.

| Model | Threshold | Accuracy | Sensitivity | Specificity | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|
| **CNN-Flatten — selected on validation** | 0.4323 | 80.29% | 99.49% | 48.29% | 0.9575 | 0.9726 |
| CNN-GAP | 0.4531 | 81.41% | 98.72% | 52.56% | 0.9479 | 0.9627 |
| MobileNetV2 | 0.2853 | 78.53% | 99.49% | 43.59% | 0.9522 | 0.9666 |
| CNN-GAP without augmentation | 0.3960 | 73.40% | 99.49% | 29.91% | 0.9288 | 0.9479 |
| CNN-GAP without class weights | 0.3491 | 76.60% | 99.23% | 38.89% | 0.9377 | 0.9582 |
| Majority-class baseline | — | 62.50% | 100.00% | 0.00% | 0.5000 | 0.6250 |

CNN-Flatten was selected using validation data, even though CNN-GAP has higher test accuracy. The test results do not change the model selection.

At its selected threshold, CNN-Flatten produces **388 true positives, 113 true negatives, 121 false positives, and 2 false negatives**. Its high sensitivity comes with limited specificity: approximately half of the normal images are classified as pneumonia. Relative to threshold 0.5, the selected threshold increases false positives from 118 to 121 without reducing the two false negatives on this test set.

The primary model's 95% group-bootstrap intervals are **98.58–100.00% for sensitivity** and **42.36–54.85% for specificity**, based on 1,000 replicates. These intervals hold the model and threshold fixed; they do not measure training or selection uncertainty.

![Test-set ROC and precision-recall curves for the five models](docs/figures/model-comparison.png)

Source files: [test metrics](artifacts/progetto/test_results.csv), [validation metrics](artifacts/progetto/validation_results.csv), [selection protocol](artifacts/progetto/frozen_protocol.json), and [confidence intervals](artifacts/progetto/primary_model_confidence_intervals.csv).

## Getting started

Use **Python 3.12** and a dedicated virtual environment. From the repository directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-progetto.txt
python -m pip install jupyterlab
python -m ipykernel install --user --name chest-xray-pneumonia --display-name "Python (Chest X-Ray)"
jupyter lab progetto.ipynb
```

On Windows, activate the environment with `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

Select the **Python (Chest X-Ray)** kernel. If you installed the requirements above, **skip the notebook's initial unpinned `!pip install ...` cell** and execute the remaining cells in order, starting with the imports and configuration. This preserves the installed dependency versions.

Internet access is needed for the first dataset and pretrained-weight downloads. The dataset is downloaded automatically; it does not need to be added to the repository. Full training runs five experiments by default and can take substantial time depending on the hardware.

### Main configuration

| Setting | Default |
|---|---:|
| Random seed | 123 |
| Image size | 180 × 180 |
| Batch size | 32 |
| Target validation recall | 0.98 |
| Maximum CNN epochs | 25 |
| MobileNetV2 head / fine-tuning epochs | 10 / 15 |
| Bootstrap replicates | 1,000 |
| Run both additional ablations | `True` |

Set `RUN_ABLATIONS=False` to run only the three main models. Early stopping can reduce the actual number of epochs. Running the notebook writes or replaces outputs under `artifacts/progetto/`.

### Environment and reproducibility

`requirements-progetto.txt` pins the environment used for functional verification, including TensorFlow 2.17.0 and NumPy 1.26.4. It **does not match the saved results' environment**. The recorded run reports the following versions in [environment.json](artifacts/progetto/environment.json):

| Package | Recorded version |
|---|---|
| TensorFlow | 2.21.0 |
| Keras | 3.15.1 |
| NumPy | 2.5.3 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| KaggleHub | 1.0.2 |

The saved tables describe that run. Fixed seeds and deterministic operations improve repeatability, but rerunning with another software stack or hardware is not guaranteed to reproduce identical partitions or metrics.

## Files and generated artifacts

```text
README.md                         Project overview and execution instructions
progetto.ipynb                    Main notebook in English
requirements-progetto.txt         Pinned dependencies for functional verification
docs/figures/model-comparison.png  ROC and precision-recall figure
artifacts/progetto/               Recorded summaries and generated outputs
```

The notebook generates models (`.keras`), training histories, split manifests, predictions, evaluation tables, and the frozen selection protocol. A lightweight GitHub release can include the notebook, dependencies, figure, and the summary CSV/JSON files linked above. Model binaries and locally generated manifests are not required to read or run the notebook. Manifests and per-image prediction files contain local filesystem paths.

## Limitations

- Evaluation is exploratory and restricted to one dataset; no external validation is provided.
- Filename-derived groups do not establish verified patient-level independence.
- Exact-duplicate checks do not detect all near-duplicates.
- A single seed does not characterize training variability or establish statistical superiority between models.
- Validation-based model and threshold selection makes validation estimates optimistic.
- Sigmoid outputs are not automatically calibrated probabilities, particularly after class-weighted training.
- Strong sensitivity does not establish clinical usefulness, especially with the observed false positive rate.

This project is intended for educational and research exploration. Its results do not establish readiness for clinical diagnosis.

## References

- [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [Keras: MobileNetV2](https://keras.io/api/applications/mobilenet/mobilenet_models/)
- [Keras: transfer learning and fine-tuning](https://keras.io/guides/transfer_learning/)
- [scikit-learn: StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [scikit-learn: decision threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html)

Dataset usage and redistribution are subject to the terms provided by the dataset publisher.
