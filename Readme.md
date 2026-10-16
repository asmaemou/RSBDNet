# RS-BDNet: Hybrid ResNet34–Swin-Tiny Network for Building Damage Assessment

RS-BDNet is an end-to-end deep-learning framework for building localization and damage-severity assessment from paired pre- and post-disaster satellite imagery.

The network combines local spatial features extracted by a Siamese ResNet-34 U-Net with global contextual features extracted by a Siamese Swin-Tiny Transformer. Bi-temporal cross-attention compares pre- and post-disaster representations, while building-guided refinement supports four-class damage-severity prediction.

RS-BDNet is independently trained and evaluated on six disaster datasets covering earthquakes, volcanic eruptions, tornadoes, hurricanes, and flooding.

> **Important:** A separate RS-BDNet model is trained for each dataset. The experiments provide a multi-dataset evaluation of the architecture, not zero-shot cross-dataset generalization.

## Overview

Given a spatially corresponding pre-disaster and post-disaster RGB image, RS-BDNet produces:

1. A binary building-localization mask.
2. A four-class building-damage map containing:
   - No damage
   - Minor damage
   - Major damage
   - Destroyed

```mermaid
flowchart LR
    A[Pre-disaster RGB image] --> C[Preprocessing]
    B[Post-disaster RGB image] --> C
    C --> D[Siamese ResNet-34 U-Net]
    C --> E[Siamese Swin-Tiny]
    D --> F[Bi-temporal cross-attention]
    E --> F
    F --> G[CNN-Transformer feature fusion]
    G --> H[Building-localization head]
    G --> I[Building-guided damage head]
    H --> J[Building-localization mask]
    H --> I
    I --> K[Four-class damage map]
    I -. Training only .-> L[Auxiliary ordinal supervision]
```

## Main Features

- **Hybrid CNN–Transformer representation:** combines local spatial details from ResNet-34 U-Net with global contextual information from Swin-Tiny.
- **Shared Siamese processing:** the same encoder weights process the pre- and post-disaster images.
- **Bi-temporal cross-attention:** compares corresponding pre- and post-disaster features in both temporal directions.
- **CNN–Transformer fusion:** integrates complementary convolutional and Transformer change features.
- **Building-guided damage prediction:** uses soft localization information to refine damage-severity features.
- **Auxiliary ordinal learning:** incorporates the natural order `no damage < minor damage < major damage < destroyed` during training.
- **Single-checkpoint inference:** each trained dataset model uses one validation-selected checkpoint rather than a large prediction ensemble.
- **Multi-dataset evaluation:** the same architecture and protocol are evaluated independently on six datasets.

## Evaluated Datasets

| Dataset | Hazard coverage | Experimental split |
|---|---|---|
| xBD | Multiple disaster types | `train + tier3` / `hold` / `test` |
| Earthquake Turkey | Earthquake | Train / validation / test |
| Mount Semeru Eruption | Volcanic eruption | Train / validation / test |
| Texas Tornadoes | Tornado | Train / validation / test |
| Hurricane Delta | Hurricane | Train / validation / test |
| Pakistan Flooding | Flood | Train / validation / test |

The xBD dataset is available from the [xView2 website](https://xview2.org/dataset).

The five event-specific datasets are derived from the Extensible Building Damage collection and are prepared in the paired-image format expected by the RS-BDNet loader.

## Repository Structure

```text
DisasterAdaptiveNet/
├── configs/                  # Configuration files
├── imgs/                     # Repository images
├── notebooks/                # Figure-generation notebooks
├── output/                   # Checkpoints, scores, and figures
├── transformer/
│   ├── sbatch/               # Slurm experiment files
│   └── scripts/              # RS-BDNet training and evaluation scripts
├── requirements.txt
├── requirements_cu128.txt
└── README.md
```

The repository also contains baseline, domain-adaptation, ablation, and Transformer experiments used during development.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/asmaemou/DisasterAdaptiveNet.git
cd DisasterAdaptiveNet
```

### 2. Create an environment

```bash
conda create -n disasteradaptivenet_cu128 python=3.10
conda activate disasteradaptivenet_cu128
```

### 3. Install PyTorch

Install a PyTorch build compatible with the CUDA version available on your system. See the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 4. Install the remaining dependencies

```bash
pip install -r requirements_cu128.txt
```

The original dependency set is available in `requirements.txt`.

## Dataset Organization

Each event-specific dataset must contain predefined training, validation, and test directories:

```text
dataset_root/
├── train/
│   ├── images/
│   └── targets/
├── val/
│   ├── images/
│   └── targets/
└── test/
    ├── images/
    └── targets/
```

Each paired sample follows this naming convention:

```text
<sample>_pre_disaster.png
<sample>_post_disaster.png
<sample>_pre_disaster_target.png
<sample>_post_disaster_target.png
```

The xBD directory is organized as:

```text
xview2/
├── train/
│   ├── images/
│   └── targets/
├── tier3/
│   ├── images/
│   └── targets/
├── hold/
│   ├── images/
│   └── targets/
└── test/
    ├── images/
    └── targets/
```

The `train` and `tier3` subsets are combined for xBD training. The `hold` subset is used for model and threshold selection, and `test` is reserved for final evaluation.

## Data Preprocessing

For each paired sample, the loader:

1. Loads the pre- and post-disaster RGB images and their masks.
2. Resizes all images and masks to `896 × 896` pixels.
3. Uses bilinear interpolation for RGB images and nearest-neighbor interpolation for masks.
4. Scales RGB values from `[0,255]` to `[0,1]`.
5. Applies ImageNet normalization:

   ```text
   Mean:     (0.485, 0.456, 0.406)
   Std. dev: (0.229, 0.224, 0.225)
   ```

6. Maps valid damage annotations to:

   ```text
   0 = no damage
   1 = minor damage
   2 = major damage
   3 = destroyed
   ```

7. Assigns the ignore value `255` outside valid building regions.

Training-only augmentation applies synchronized horizontal flips, vertical flips, and rotations by `0°`, `90°`, `180°`, or `270°` to the paired images and masks. Random augmentation is not applied during validation or testing.

## Architecture

### Siamese ResNet-34 U-Net

The ResNet-34 U-Net branch extracts local spatial features and preserves building-boundary information. Its weights are shared between the two temporal inputs.

### Siamese Swin-Tiny

The Swin-Tiny branch extracts hierarchical contextual features through shifted-window self-attention. Its weights are also shared between the pre- and post-disaster inputs.

### Bi-Temporal Cross-Attention

Windowed cross-attention is applied in both directions: post-disaster features attend to pre-disaster features, and pre-disaster features attend to post-disaster features. This highlights relevant temporal changes while bounding attention memory.

### CNN–Transformer Fusion

Change features from the ResNet and Swin branches are concatenated and refined into a shared hybrid representation.

### Building-Guided Damage Prediction

The localization head predicts building presence. Its soft probability map emphasizes fused features in likely building regions before four-class damage prediction.

### Auxiliary Ordinal Learning

The ordinal head produces three cumulative predictions:

```text
damage > no damage
damage > minor damage
damage > major damage
```

The ordinal head is used only for auxiliary supervision during training. Inference uses the primary localization and four-class damage heads.

## Training Objective

The localization objective combines binary cross-entropy and Dice losses:

```text
Lloc = LBCE + LDice,loc
```

The damage objective combines focal, multiclass Dice, and auxiliary ordinal losses:

```text
Ldamage = LFocal,damage + LDice,damage + 0.30 Lordinal
```

The complete objective is:

```text
Ltotal = LBCE + LDice,loc + LFocal,damage + LDice,damage + 0.30 Lordinal
```

Class weights are calculated from the training data. Pixels labeled `255` are excluded from the damage loss.

## Training Configuration

| Setting | Value |
|---|---:|
| Input resolution | `896 × 896` |
| Maximum epochs | `60` |
| Batch size | `2` |
| Optimizer | AdamW |
| Initial learning rate | `2 × 10⁻⁵` |
| Weight decay | `1 × 10⁻⁴` |
| Gradient clipping | Maximum norm `1.0` |
| Early-stopping patience | `12` epochs |
| Random seed | `42` |
| Ordinal-loss weight | `0.30` |
| Mixed precision | Enabled with `--amp` |

The ResNet-34 and Swin-Tiny encoders are initialized with ImageNet-pretrained weights.

## Training with Slurm

The supplied Slurm files contain cluster-specific paths. Update `PROJECT_ROOT`, `DATA_ROOT` or `XBD_ROOT`, and `OUTPUT_DIR` before submitting them on another system.

### xBD

```bash
sbatch transformer/sbatch/train_xbd_bitemporal_building_crossattention_ordinal.sbatch
```

### Earthquake Turkey

```bash
sbatch transformer/sbatch/train_earthquake_turkey_bitemporal_building_crossattention_ordinal.sbatch
```

### Mount Semeru Eruption

```bash
sbatch transformer/sbatch/train_mount_semeru_bitemporal_building_crossattention_ordinal.sbatch
```

### Texas Tornadoes

```bash
sbatch transformer/sbatch/train_texas_tornadoes_bitemporal_building_crossattention_ordinal.sbatch
```

### Hurricane Delta

```bash
sbatch transformer/sbatch/train_hurricane_delta_bitemporal_building_crossattention_ordinal.sbatch
```

### Pakistan Flooding

```bash
sbatch transformer/sbatch/train_pakistan_flooding_bitemporal_building_crossattention_ordinal.sbatch
```

Monitor jobs with:

```bash
squeue -u "$USER"
```

Follow a job log with:

```bash
tail -f transformer/sbatch/logs/<job-name>_<job-id>.out
```

## Model and Threshold Selection

After every epoch, the model is evaluated on the validation partition. The localization threshold is selected from:

```text
{0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75}
```

The validation score is:

```text
Sval = 0.3 F1loc + 0.7 F1damage,H
```

where `F1loc` is the building-localization F1 score and `F1damage,H` is the harmonic mean of the four class-wise damage F1 scores.

The checkpoint and threshold with the highest validation score are retained. Early stopping is applied after 12 consecutive epochs without improvement. The selected checkpoint and threshold are then applied to the held-out test partition.

## Experiment Outputs

A typical experiment directory contains:

```text
experiment_output/
├── checkpoints/
│   ├── best.pt
│   ├── last.pt
│   └── epoch_*.pt
├── scores/
│   ├── epoch_*_hold_threshold_scan.csv
│   ├── scores_xbd_test.json
│   └── scores_xbd_test.txt
└── history.json
```

Some event-specific experiments retain `xbd` in score filenames because they reuse the same evaluation infrastructure.

## Qualitative Results

After training all six models, generate the qualitative figures with:

```bash
sbatch transformer/sbatch/generate_rsbdnet_qualitative_results.sbatch
```

The generator loads each validation-selected `best.pt` checkpoint and evaluates examples from the held-out test partitions. Examples are selected using ground-truth severity coverage only; prediction quality is not used during selection.

Results are written to:

```text
output/RSBDNet_qualitative_results/
```

The directory contains per-dataset and combined PNG, PDF, and SVG figures, along with an auditable sample-selection manifest and metadata file.

## Jupyter Notebooks

- `notebooks/building_damage_class_distribution.ipynb` — visualizes class-wise damage distributions.
- `notebooks/regenerate_qualitative_overview_large_fonts.ipynb` — rebuilds the qualitative overview with larger typography without changing predictions.
- `notebooks/rsbdnet_simple_architecture.ipynb` — generates a simplified architecture workflow.

## Reproducibility

- The random seed is fixed to `42`.
- Training, validation, and test partitions are kept separate.
- Checkpoint and threshold selection use validation data only.
- Test data are reserved for final evaluation.
- The same preprocessing and training settings are applied across the six independently trained models.
- Qualitative samples are selected without considering prediction quality.
- The ordinal head is used only during training.
- Each dataset model uses one validation-selected checkpoint for inference.

## Parameter and Checkpoint Analysis

Parameter-counting utilities are provided in:

```text
transformer/scripts/count_hbg_caon_parameters.py
transformer/scripts/count_xview2_winner_parameters.py
```

RS-BDNet contains approximately `55.3 million` trainable parameters and requires one checkpoint for each independently trained dataset model.

The xView2 comparisons examine publicly released checkpoint collections from the first-, second-, and third-place competition solutions. This comparison concerns parameter and checkpoint complexity and should not be interpreted as a latency, memory, or energy comparison unless those quantities are evaluated separately.

## Citation

If you use RS-BDNet, please cite the associated manuscript after publication. A temporary citation entry is:

```bibtex
@article{mouradi2026rsbdnet,
  author  = {Mouradi, Asmae and Kshirsagar, Shruti},
  title   = {Parameter-Efficient Multi-Hazard Building Damage Assessment
             Using a Hybrid ResNet34--Swin-Tiny Network},
  year    = {2026},
  note    = {Manuscript}
}
```

This repository builds upon the original DisasterAdaptiveNet implementation:

```bibtex
@article{hafner2025disasteradaptivenet,
  title   = {DisasterAdaptiveNet: A Robust Network for Multi-Hazard
             Building Damage Detection from Very-High-Resolution
             Satellite Imagery},
  author  = {Hafner, Sebastian and Gerard, Sebastian and
             Sullivan, Josephine and Ban, Yifang},
  journal = {International Journal of Applied Earth Observation
             and Geoinformation},
  volume  = {143},
  pages   = {104756},
  year    = {2025},
  doi     = {10.1016/j.jag.2025.104756}
}
```

## Acknowledgments

This repository is derived from the original [DisasterAdaptiveNet](https://github.com/SebastianHafner/DisasterAdaptiveNet) project by Hafner et al.

The repository also contains adaptations and evaluation utilities associated with publicly available xView2 benchmark solutions. The original authors and repositories retain credit for those implementations.

## License and Third-Party Components

This repository contains code derived from or adapted from multiple research implementations. Consult the licenses included in the corresponding upstream and baseline directories before redistribution or commercial use.

Dataset licenses and usage conditions are determined by their respective providers.
