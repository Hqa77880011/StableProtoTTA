# StableProtoTTA

Implementation of **StableProtoTTA: Reliability-Gated Online Prototype Alignment for Stable Test-Time Adaptation**.

## Installation

Use Python `>=3.10`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

`prepare` downloads [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). Download [CIFAR-10-C](https://zenodo.org/records/2535967) separately and extract `labels.npy` and corruption arrays under `data/CIFAR-10-C` (config key `corruption_root`). For CIFAR-100, use `configs/cifar100.json` and [CIFAR-100-C](https://zenodo.org/records/3555552).

```bash
python -m stableprototta prepare --config configs/cifar10.json
```

## Training and execution

`configs/cifar10.json` fixes seed `2026`. Use `--seed` for a new run and keep its data/config paired. Paths are configurable with the CLI or JSON.

```bash
python -m stableprototta train --config configs/cifar10.json --output outputs/source
```

## Evaluation

```bash
python -m stableprototta evaluate --config configs/cifar10.json --checkpoint outputs/source/source.pt --output outputs/source
```

## Comparisons and ablations

`compare` runs the configured variants; it includes model fitting where required. Keep comparison outputs separate from the main run.

```bash
python -m stableprototta compare --config configs/cifar10.json --checkpoint outputs/source/source.pt --output outputs/comparison
python -m stableprototta plot --config configs/cifar10.json --output outputs/comparison
```

Use `--protocol continual` for persistent state across severity-five domains; `order_seeds` define shared deterministic corruption orders. Episodic evaluation resets model, optimizer, threshold and memory before each corruption/severity pair. Target labels are passed only to post-hoc statistics, never to `Adapter.adapt`.

`configs/continual_controls.json` runs the five matched gate/loss/memory arms on CIFAR-100-C, including `confidence_memory` (full loss gate but a confidence-only memory mask). `configs/imbalance.json` fixes class IDs 80--99 at 10 samples and IDs 0--79 at 100 samples per domain, with selection fixed by `sampling_seed` before unlabeled adaptation. These configurations reuse the same source checkpoint:

```bash
python -m stableprototta compare --config configs/continual_controls.json --checkpoint outputs/source/source.pt --output outputs/continual
python -m stableprototta compare --config configs/imbalance.json --checkpoint outputs/source/source.pt --output outputs/imbalance
```

Train the source with `configs/cifar100.json` for these CIFAR-100 controls. Supported baselines are Source, BN and Tent, plus the paper's gate/prototype component controls. EATA, T3A and CoTTA integrations are not supplied; no differently defined surrogate is labeled as one of those methods.

## Outputs

Run directories contain `source.pt`, `history.csv`, `domains.csv`, `stream.csv`, `metrics.json` as produced by the corresponding steps. `config.json` records effective settings. Comparison results are in `outputs/comparison/comparison.csv`; plotting reads those computed files.

## Verification

Rechecked on 2026-11-01 by running the existing `python tests/smoke.py` end to end, with one CPU thread and the source package available on `PYTHONPATH=src`.

```bash
python tests/smoke.py
```

Passed on Windows / Python 3.13.5 / NumPy 2.1.3 (OpenBLAS) / PyTorch 2.7.1+cu118 / torchvision 0.22.1: two source-training steps, BN-only finite gradients and actual updates, pre-update prediction recording, rejection with unchanged model/optimizer/prototypes, complete adaptive-state save/load and continuation, episodic/continual array loading, plotting and hand pooled/class metrics. Synthetic fixtures and permissive test thresholds only exercise functionality; they are not CIFAR-C results. Full source training, the 75-domain benchmarks and the paper's accuracy values were not rerun.

| Paper module or metric | Code entry | Configuration or command |
|---|---|---|
| CIFAR-stem ResNet-18 and source checkpoint | `common.FeatureNet`, `cli.train` | `train --config configs/cifar100.json` |
| Clipped quantile/EMA confidence and prototype agreement gate | `method.Adapter.adapt` | `quantile`, threshold bounds/momentum, `gamma` |
| Entropy + prototype alignment; BN affine only | `Adapter.adapt` | `tta_lr=0.001`, `lambda_prototype=0.2` |
| Shared detached confidence-weighted memory / normalization | `Adapter.adapt`, `Adapter.normalize` | momentum `0.95`, additive `epsilon=1e-6` |
| Full causal adaptive state | `Adapter.state_dict/load_state_dict` | `adapted.pt` contains the last order's final state |
| 75-domain episodic and 3-order continual accuracy | `cli.evaluate` | `protocol`, `eval_corruptions`, `severities`, `order_seeds` |
| Acceptance and pooled accepted precision per order | `metrics.order_metrics` | Counts in `domains.csv`; `per_order` in `metrics.json` |
| Rare/common, class-macro, exposure-micro accuracy | `metrics.order_metrics`, `common.CorruptionImages` | `configs/imbalance.json` |

Accuracy is in percent; acceptance and accepted precision are fractions. Domain accuracies are equally weighted within each order, then orders are equally weighted. Accepted precision is a ratio of pooled correct/accepted counts within an order, not an average of batch precisions. With no accepted sample it is null. Only classes present in the fixed stream enter macro means. The source/adapter infrastructure predates this update; this update adds the shared-mask control, additive-epsilon normalization, full-state restore and explicit order/group aggregation. Original paper results remain separate from this implementation verification.
