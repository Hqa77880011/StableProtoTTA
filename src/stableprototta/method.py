import torch
from copy import deepcopy
from torch import nn

VARIANTS = [
    "stableprototta",
    "source",
    "bn",
    "tent",
    "gate",
    "dynamic_gate",
    "proto",
    "no_proto_gate",
    "no_proto_loss",
    "no_memory",
    "confidence_memory",
]


class Adapter:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.variant = config["variant"]
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {self.variant}")
        self.model.eval()
        self.model.requires_grad_(False)
        parameters = []
        if self.variant != "source":
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.train()
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None
                    if module.affine:
                        module.weight.requires_grad_(True)
                        module.bias.requires_grad_(True)
                        parameters.extend([module.weight, module.bias])
        self.optimizer = (
            torch.optim.Adam(parameters, lr=config["tta_lr"]) if parameters else None
        )
        self.prototypes = self.normalize(model.classifier.weight.detach().clone(), dim=1)
        self.threshold = None

    def normalize(self, value, dim):
        """Use the paper's additive-epsilon normalization, including near zero."""
        return value / (value.norm(dim=dim, keepdim=True) + self.config.get("epsilon", 1e-6))

    def state_dict(self):
        """Snapshot all persistent adaptation state for an exact continuation."""
        return deepcopy({"model": self.model.state_dict(), "prototypes": self.prototypes,
                         "threshold": self.threshold,
                         "optimizer": self.optimizer.state_dict() if self.optimizer else None})

    def load_state_dict(self, state):
        """Restore a snapshot into an adapter with the same model and settings."""
        self.model.load_state_dict(state["model"])
        self.prototypes = state["prototypes"].to(next(self.model.parameters()).device).clone()
        self.threshold = state["threshold"]
        if self.optimizer is not None and state["optimizer"] is not None:
            self.optimizer.load_state_dict(state["optimizer"])

    def adapt(self, x):
        logits, features = self.model(x)
        predictions = logits.detach().argmax(1)
        confidence = logits.detach().softmax(1).max(1).values
        normalized = self.normalize(features, dim=1)
        similarity = (normalized.detach() * self.prototypes[predictions]).sum(1)
        variant = self.variant
        if variant in ["source", "bn"]:
            return (
                predictions,
                torch.zeros(len(x), dtype=torch.bool, device=x.device),
                {"threshold": 0.0, "loss": 0.0},
            )
        dynamic = variant in [
            "stableprototta",
            "dynamic_gate",
            "no_proto_gate",
            "no_proto_loss",
            "no_memory",
            "confidence_memory",
        ]
        if dynamic:
            quantile = torch.quantile(confidence, self.config["quantile"]).item()
            candidate = (
                quantile
                if self.threshold is None
                else self.config["threshold_momentum"] * self.threshold
                + (1 - self.config["threshold_momentum"]) * quantile
            )
            self.threshold = min(
                self.config["threshold_max"],
                max(self.config["threshold_min"], candidate),
            )
        else:
            self.threshold = self.config["fixed_threshold"]
        mask = torch.ones(len(x), dtype=torch.bool, device=x.device)
        if variant not in ["tent", "proto"]:
            mask &= confidence >= self.threshold
        confidence_mask = mask.clone()
        if variant in ["stableprototta", "no_proto_loss", "no_memory", "confidence_memory"]:
            mask &= similarity >= self.config["gamma"]
        loss_value = 0.0
        accepted = normalized.detach().clone()
        if mask.any():
            probabilities = logits.softmax(1)
            loss = (
                -(probabilities[mask] * probabilities[mask].clamp_min(1e-8).log())
                .sum(1)
                .mean()
            )
            if variant in ["stableprototta", "proto", "no_proto_gate", "no_memory", "confidence_memory"]:
                alignment = 1 - (
                    normalized[mask] * self.prototypes[predictions[mask]].detach()
                ).sum(1)
                loss = loss + self.config["lambda_prototype"] * alignment.mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            loss_value = loss.item()
        memory_mask = confidence_mask if variant == "confidence_memory" else mask
        if variant in ["stableprototta", "proto", "no_proto_gate", "no_proto_loss", "confidence_memory"]:
            with torch.no_grad():
                for label in predictions[memory_mask].unique():
                    subset = memory_mask & (predictions == label)
                    center = (accepted[subset] * confidence[subset, None]).sum(0) / (
                        confidence[subset].sum() + self.config.get("epsilon", 1e-6))
                    self.prototypes[label] = self.normalize(
                        self.config["prototype_momentum"] * self.prototypes[label]
                        + (1 - self.config["prototype_momentum"]) * center, dim=0)
        return predictions, mask, {"threshold": self.threshold, "loss": loss_value}
