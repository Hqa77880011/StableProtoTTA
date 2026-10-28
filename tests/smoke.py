"""Check test-then-adapt, BN-only gradients, rejected batches and stream metrics."""

from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from stableprototta.cli import train, evaluate, source_model
from stableprototta.common import prepare, plot_csv, CorruptionImages
from stableprototta.method import Adapter
from stableprototta.metrics import order_metrics


def main():
    torch.set_num_threads(1)
    out = Path("outputs/smoke").resolve()
    rng = np.random.default_rng(71)
    for split in ["train", "val", "test"]:
        for label in ["a", "b"]:
            path = out / "raw" / split / label
            path.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path / f"{i}.png")
    config = json.loads(Path("configs/cifar10.json").read_text())
    config.update(dataset="folder", root=str(out / "raw"), data=str(out / "data"),
                  output=str(out), device="cpu", epochs=1, max_steps=2, batch_size=2,
                  eval_corruptions=["noise", "blur"], severities=[1, 5],
                  order_seeds=[11, 23], corruption_root=str(out / "corruptions"),
                  threshold_min=0., threshold_max=0., gamma=-1.)
    prepare(config)
    checkpoint = train(config)
    source = torch.load(checkpoint, weights_only=False)
    corruption_root = Path(config["corruption_root"])
    corruption_root.mkdir(parents=True, exist_ok=True)
    np.save(corruption_root / "labels.npy", np.array([0, 1]))
    for name in config["eval_corruptions"]:
        np.save(corruption_root / f"{name}.npy", rng.integers(0, 256, (10, 32, 32, 3), dtype=np.uint8))
    adapter = Adapter(source_model(source, config), config)
    x = torch.rand(2, 3, 32, 32)
    before = deepcopy(adapter.model.state_dict())
    with torch.no_grad():
        expected = adapter.model(x)[0].argmax(1)
    prediction, mask, detail = adapter.adapt(x)
    assert torch.equal(prediction, expected) and mask.all() and np.isfinite(detail["loss"])
    bn_names = {name for name, p in adapter.model.named_parameters() if p.requires_grad}
    assert bn_names and all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.model.parameters() if p.requires_grad)
    assert any(not torch.equal(before[name], p) for name, p in adapter.model.named_parameters() if name in bn_names)
    assert all(torch.equal(before[name], p) for name, p in adapter.model.named_parameters() if name not in bn_names)
    state = adapter.state_dict()
    torch.save(state, out / "adapter.pt")
    restored = Adapter(source_model(source, config), config)
    restored.load_state_dict(torch.load(out / "adapter.pt", weights_only=False))
    with torch.no_grad():
        torch.testing.assert_close(adapter.model(x)[0], restored.model(x)[0])
    a, m, _ = adapter.adapt(x)
    b, n, _ = restored.adapt(x)
    assert torch.equal(a, b) and torch.equal(m, n)
    torch.testing.assert_close(adapter.prototypes, restored.prototypes)
    rejected = Adapter(source_model(source, config), dict(config, threshold_min=1., threshold_max=1.))
    old = rejected.state_dict()
    _, mask, _ = rejected.adapt(x)
    assert not mask.any() and rejected.threshold == 1.
    assert all(torch.equal(v, rejected.model.state_dict()[k]) for k, v in old["model"].items())
    assert torch.equal(old["prototypes"], rejected.prototypes) and not rejected.optimizer.state
    confidence_memory = Adapter(source_model(source, config), dict(config, variant="confidence_memory", gamma=1.))
    old = confidence_memory.state_dict()
    _, mask, _ = confidence_memory.adapt(x)
    assert not mask.any() and not torch.equal(old["prototypes"], confidence_memory.prototypes)
    assert all(torch.equal(v, confidence_memory.model.state_dict()[k]) for k, v in old["model"].items())
    for protocol in ["episodic", "continual"]:
        summary, rows = evaluate(dict(config, checkpoint=checkpoint, protocol=protocol, output=str(out / protocol)))
        assert summary["domains_per_order"] == (4 if protocol == "episodic" else 2)
        assert summary["accepted_ratio"] == 1.
    plot_csv(out / "continual/stream.csv", out / "stream.pdf", "batch", "accuracy")
    measured = order_metrics([
        dict(accuracy=50, samples=2, accepted=1, accepted_correct=1),
        dict(accuracy=75, samples=4, accepted=3, accepted_correct=2),
    ], [1, 3], [2, 4], [0])
    assert measured["accepted_precision"] == .75 and measured["rare_accuracy"] == 50 and measured["macro_accuracy"] == 62.5
    assert abs(measured["micro_accuracy"] - 100 * 4 / 6) < 1e-10
    # Equal output when no class is accepted and pooled precision are separate contracts.
    sample = CorruptionImages(corruption_root, "noise", 1, class_counts={"0": 1, "1": 1})
    assert set(sample.labels) == {0, 1}
    print("PASS: source 2-step training, test-then-adapt, BN-only gradients/updates, full-state restore/continuation, no-acceptance skip, episodic/continual streams, pooled and class metrics")


if __name__ == "__main__":
    main()
