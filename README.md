# StableProtoTTA

Implementation of **StableProtoTTA: Reliability-Gated Online Prototype Alignment for Stable Test-Time Adaptation**. Confidence and predicted-class prototype similarity select one shared subset for entropy minimization, feature alignment and prototype-memory updates. Only batch-normalization affine parameters adapt.

## Installation

Use Python 3.10 or newer. Install matching PyTorch and torchvision builds for your device, then install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. For NVIDIA GPUs, choose a CUDA build from the [PyTorch installer](https://pytorch.org/get-started/locally/) in place of the CPU command. The supported dependency range is PyTorch 2.4–2.7 with its matching torchvision release.

## Prepare data and train the source model

`prepare` downloads [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) through torchvision and creates a seeded, class-balanced validation split of 5,000 training images. The remaining 45,000 images form the training pool; the official test split is untouched. The prepared directory contains `train.npz`, `val.npz`, `test.npz` and `meta.json`. Array indices identify the original training examples. A custom image collection can use `dataset="folder"` with matching class directories under `root/train`, `root/val` and `root/test`.

Download and extract [CIFAR-10-C](https://zenodo.org/records/2535967) or [CIFAR-100-C](https://zenodo.org/records/3555552). Set `corruption_root` to the directory containing `labels.npy` and the corruption arrays. Each corruption array has five consecutive severity blocks. These arrays are memory mapped during evaluation. They are excluded from training and validation.

```bash
python -m stableprototta prepare --config configs/cifar10.json
python -m stableprototta train --config configs/cifar10.json --output outputs/source
```

The clean source checkpoint is saved as `outputs/source/source.pt`. All methods in a comparison consume this same checkpoint.

## Run adaptation and comparisons

```bash
python -m stableprototta evaluate --config configs/cifar10.json --checkpoint outputs/source/source.pt --output outputs/episodic
python -m stableprototta plot --config configs/cifar10.json --output outputs/episodic
python -m stableprototta compare --config configs/cifar10.json --checkpoint outputs/source/source.pt --output outputs/comparison
python -m stableprototta evaluate --config configs/cifar10.json --protocol continual --checkpoint outputs/source/source.pt --output outputs/continual
```

Episodic evaluation restores the source model, optimizer, threshold and prototypes before every corruption–severity domain. Continual evaluation uses severity five, shuffles the 15 corruption types with each seed in `order_seeds`, and resets only at the start of an order. Accuracy averages domains within each order and then averages orders.

| Variant | Adaptation |
| --- | --- |
| `source` | Frozen source model and running BN statistics |
| `bn` | Current-batch BN statistics with no gradient update |
| `tent` | BN-affine entropy minimization on all samples |
| `gate` | Fixed confidence filter |
| `dynamic_gate` | Smoothed quantile confidence filter |
| `proto` | Entropy and prototype alignment on all samples |
| `stableprototta` | Joint gate, alignment and accepted-feature memory |
| `no_proto_gate` | Removes the cosine filter |
| `no_proto_loss` | Removes alignment but keeps gated memory |
| `no_memory` | Keeps classifier-initialized prototypes fixed |

These are the implemented executable comparisons; EATA, T3A and CoTTA are not entries in the supplied suite.

## Parameters and outputs

Commands read the supplied JSON configuration. `--data`, `--output`, `--device`, `--seed`, `--batch-size`, `--epochs`, `--max-steps` and `--limit` override individual fields. `--device auto` chooses CUDA when available. `max_steps=0` and `limit=0` mean no limit. `max_steps` stops optimization after the requested number of updates; it does not produce a trained benchmark model. For other fields use `--set KEY=JSON`, for example `--set lr=0.01` or `--set tta_lr=0.001`. String values require JSON quotes.

The paper settings are `tta_lr=0.001`, batch size 64, confidence quantile 0.70, threshold momentum 0.90, threshold bounds [0.55, 0.95], cosine gate 0.35, prototype momentum 0.95 and alignment weight 0.20. One Adam update is made for each nonempty accepted subset.

`domains.csv` records domain accuracy, accepted ratio and accepted pseudo-label precision. `stream.csv` records causal batch predictions, threshold and loss. Ground-truth labels are accessed only after adaptation to compute these measurements. `metrics.json` stores the aggregate score and sample standard deviation across orders. `adapted.pt` stores the final BN/model, optimizer, prototype and threshold state. `stream.pdf` shows per-batch accuracy.

## State and prediction order

Predictions are recorded from the forward pass before any optimizer update. The current confidence quantile updates the threshold; selection compares features with the preceding prototype memory. The same mask controls both losses and memory. If no sample is accepted, the optimizer and prototypes remain unchanged while the confidence threshold still advances. Prototypes start from normalized source-classifier rows. BN uses current-batch statistics without accumulating running statistics. The persistent memory is class-sized and the similarity check compares only the predicted-class prototype.

Data, weights and generated outputs are ignored by Git. Run directories store the effective `config.json`, CSV measurements and JSON metrics. Plots read the generated measurement files.
