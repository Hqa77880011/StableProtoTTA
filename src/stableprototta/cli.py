from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.nn import functional as F

from .common import (
    Images,
    CorruptionImages,
    FeatureNet,
    loader,
    accuracy,
    device_of,
    seed_everything,
    prepare,
    parse_config,
    save_json,
    save_rows,
    plot_csv,
)
from .method import Adapter, VARIANTS
from .metrics import order_metrics
from .baselines import BASELINES, BaselineAdapter, source_fisher


def train(config):
    seed_everything(config["seed"])
    dataset = Images(config, "train", augment=True)
    model = FeatureNet(len(dataset.classes)).to(device_of(config))
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=0.9,
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
    history = []
    steps = 0
    for epoch in range(config["epochs"]):
        model.train()
        loss_sum = count = 0
        for x, y in loader(dataset, config, shuffle=True):
            x, y = x.to(device_of(config)), y.to(device_of(config))
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x)[0], y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            count += len(y)
            steps += 1
            if config["max_steps"] and steps >= config["max_steps"]:
                break
        row = {
            "epoch": epoch + 1,
            "variant": "source",
            "loss": loss_sum / count,
            "val_accuracy": accuracy(model, Images(config, "val"), config),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        scheduler.step()
        if config["max_steps"] and steps >= config["max_steps"]:
            break
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "classes": dataset.classes, "config": config},
        output / "source.pt",
    )
    save_rows(output / "history.csv", history)
    save_json(output / "config.json", config)
    return str(output / "source.pt")


def source_model(state, config):
    model = FeatureNet(len(state["classes"])).to(device_of(config))
    model.load_state_dict(state["model"])
    return model


def evaluate(config):
    state = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    if config["protocol"] not in ["episodic", "continual"]:
        raise ValueError("protocol must be episodic or continual")
    if not config.get("corruption_root"):
        raise ValueError("Set corruption_root to the extracted CIFAR-C directory")
    domains = [
        (name, severity)
        for name in config["eval_corruptions"]
        for severity in config["severities"]
    ]
    orders = (
        config["order_seeds"] if config["protocol"] == "continual" else [config["seed"]]
    )
    fishers = None
    if config["variant"] == "eata":
        fishers = source_fisher(
            source_model(state, config), Images(config, "train"),
            config.get("fisher_samples", 2000),
        )
    rows, batches, per_order = [], [], []
    for order_seed in orders:
        sequence = list(domains)
        if config["protocol"] == "continual":
            sequence = [
                (name, config["continual_severity"])
                for name in config["eval_corruptions"]
            ]
            random.Random(order_seed).shuffle(sequence)
        adapter = None
        order_start = len(rows)
        class_correct = np.zeros(len(state["classes"]), dtype=np.int64)
        class_counts = np.zeros_like(class_correct)
        batch_index = 0
        seed_everything(order_seed)
        for name, severity in sequence:
            if adapter is None or config["protocol"] == "episodic":
                if config["variant"] in BASELINES:
                    adapter = BaselineAdapter(source_model(state, config), config, fishers)
                else:
                    adapter = Adapter(source_model(state, config), config)
            data = CorruptionImages(
                config["corruption_root"], name, severity, config["limit"],
                config.get("class_counts"), config.get("sampling_seed", 2026)
            )
            correct = total = accepted = accepted_correct = 0
            for x, y in loader(data, config):
                prediction, mask, details = adapter.adapt(x.to(device_of(config)))
                prediction, mask = prediction.cpu(), mask.cpu()
                hits = prediction == y
                class_counts += np.bincount(y.numpy(), minlength=len(class_counts))
                class_correct += np.bincount(y[hits].numpy(), minlength=len(class_counts))
                correct += hits.sum().item()
                total += len(y)
                accepted += mask.sum().item()
                accepted_correct += (hits & mask).sum().item()
                batch_index += 1
                batches.append(
                    {
                        "variant": config["variant"],
                        "order_seed": order_seed,
                        "batch": batch_index,
                        "corruption": name,
                        "severity": severity,
                        "accuracy": 100 * hits.float().mean().item(),
                        "accepted": mask.sum().item(),
                        "samples": len(y),
                        **details,
                    }
                )
            rows.append(
                {
                    "variant": config["variant"],
                    "protocol": config["protocol"],
                    "order_seed": order_seed,
                    "corruption": name,
                    "severity": severity,
                    "accuracy": 100 * correct / total,
                    "accepted_ratio": accepted / total,
                    "samples": total,
                    "correct": correct,
                    "accepted": accepted,
                    "accepted_correct": accepted_correct,
                    "accepted_precision": accepted_correct / accepted
                    if accepted
                    else None,
                }
            )
        per_order.append({"order_seed": order_seed, **order_metrics(rows[order_start:], class_correct, class_counts, config.get("rare_classes", []))})
    order_scores = [
        np.mean([row["accuracy"] for row in rows if row["order_seed"] == seed])
        for seed in orders
    ]
    summary = {
        "variant": config["variant"],
        "protocol": config["protocol"],
        "accuracy": float(np.mean(order_scores)),
        "order_std": float(np.std(order_scores, ddof=1))
        if len(order_scores) > 1
        else 0.0,
        "domains_per_order": len(sequence),
        "per_order": per_order,
    }
    for key in ["micro_accuracy", "macro_accuracy", "rare_accuracy", "common_accuracy", "accepted_ratio", "accepted_precision"]:
        values = [r[key] for r in per_order if r[key] is not None]
        summary[key] = float(np.mean(values)) if values else None
    output = Path(config["output"])
    save_rows(output / "domains.csv", rows)
    save_rows(output / "stream.csv", batches)
    save_json(output / "metrics.json", summary)
    save_json(output / "config.json", config)
    torch.save(
        adapter.state_dict(),
        output / "adapted.pt",
    )
    print(json.dumps(summary), flush=True)
    return summary, rows


def compare(config):
    summaries, rows = [], []
    for variant in config["variants"]:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        current = dict(
            config, variant=variant, output=str(Path(config["output"]) / variant)
        )
        summary, domains = evaluate(current)
        summaries.append(summary)
        rows.extend(domains)
    save_rows(Path(config["output"]) / "comparison.csv", summaries)
    save_rows(Path(config["output"]) / "domains.csv", rows)


def main():
    command, config = parse_config(["prepare", "train", "evaluate", "compare", "plot"])
    if command == "prepare":
        prepare(config)
    elif command == "train":
        train(config)
    elif command == "evaluate":
        evaluate(config)
    elif command == "compare":
        compare(config)
    else:
        root = Path(config["output"])
        if (root / "stream.csv").exists():
            plot_csv(root / "stream.csv", root / "stream.pdf", "batch", "accuracy")
        elif (root / "domains.csv").exists():
            plot_csv(
                root / "domains.csv", root / "severity.pdf", "severity", "accuracy"
            )
        else:
            plot_csv(
                root / "history.csv", root / "training.pdf", "epoch", "val_accuracy"
            )
