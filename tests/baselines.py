"""Tiny baseline mechanics, independent of CIFAR-C benchmark scores."""

from copy import deepcopy
from pathlib import Path
import torch
from stableprototta.baselines import BaselineAdapter, source_fisher
from stableprototta.cli import evaluate, source_model
from stableprototta.common import Images


def main():
    torch.set_num_threads(1)
    out = Path("outputs/smoke").resolve()
    checkpoint = out / "source.pt"
    if not checkpoint.exists():
        raise RuntimeError(
            "Run python tests/smoke.py first to create the file-backed fixture"
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(
        state["config"],
        checkpoint=str(checkpoint),
        device="cpu",
        t3a_filter_k=2,
        fisher_samples=2,
        eata_margin_factor=2.0,
        eata_diversity=1.1,
        cotta_augmentations=2,
        cotta_anchor_threshold=1.1,
        cotta_restore=0.0,
        protocol="episodic",
        eval_corruptions=["noise"],
        severities=[1],
    )
    dataset = Images(cfg, "train")
    x = torch.stack([dataset[i][0] for i in range(2)])
    fishers = source_fisher(source_model(state, cfg), dataset, 2)
    assert all(torch.isfinite(pair[0]).all() for pair in fishers.values())
    assert sum(float(pair[0].sum()) for pair in fishers.values()) > 0
    for variant in ("eata", "t3a", "cotta"):
        current = dict(cfg, variant=variant, output=str(out / variant))
        torch.manual_seed(71)
        adapter = BaselineAdapter(
            source_model(state, current),
            current,
            fishers if variant == "eata" else None,
        )
        before = deepcopy(adapter.model.state_dict())
        predictions, mask, details = adapter.adapt(x)
        assert predictions.shape == (2,) and torch.isfinite(
            torch.tensor(details["loss"])
        )
        if variant != "t3a":
            gradients = [
                parameter.grad
                for parameter in adapter.model.parameters()
                if parameter.requires_grad
            ]
            assert all(
                value is not None and torch.isfinite(value).all() for value in gradients
            )
            assert sum(float(value.abs().sum()) for value in gradients) > 0
            assert any(
                not torch.equal(before[name], value)
                for name, value in adapter.model.state_dict().items()
                if name in before
            )
        else:
            assert len(adapter.supports) <= 2 * len(state["classes"])
        snapshot = adapter.state_dict()
        file = out / (variant + "_state.pt")
        torch.save(snapshot, file)
        # Continuation restores algorithm buffers, optimizer and CoTTA randomness.
        expected = adapter.adapt(x)[0]
        restored = BaselineAdapter(
            source_model(state, current),
            current,
            fishers if variant == "eata" else None,
        )
        restored.load_state_dict(
            torch.load(file, map_location="cpu", weights_only=False)
        )
        observed = restored.adapt(x)[0]
        torch.testing.assert_close(expected, observed, rtol=0, atol=0)
        summary, rows = evaluate(current)
        assert len(rows) == 1 and 0 <= summary["accuracy"] <= 100
    print(
        "PASS: source-only Fisher, EATA and CoTTA gradients/updates, bounded T3A memory, baseline state save/load/continuation, file-backed evaluation"
    )


if __name__ == "__main__":
    main()
