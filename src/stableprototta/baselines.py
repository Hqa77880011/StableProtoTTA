"""Single-process EATA, T3A and CoTTA adaptations with the shared source model.

Algorithm sources and implementation conventions are listed in README.
No target labels are accepted by these interfaces.
"""

from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

BASELINES = ("eata", "t3a", "cotta")


def entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def configure_bn(model, all_parameters=False):
    model.eval().requires_grad_(all_parameters)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train().requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = module.running_var = None
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def source_fisher(model, dataset, samples):
    """Average per-image pseudo-label score-gradient squares on clean source only."""
    if samples < 1:
        raise ValueError("EATA requires a positive clean-source fisher_samples")
    model.eval().requires_grad_(False)
    parameters = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            for key in ("weight", "bias"):
                parameter = getattr(module, key)
                parameter.requires_grad_(True)
                parameters[name + "." + key] = parameter
    fisher = {
        name: [torch.zeros_like(parameter), parameter.detach().clone()]
        for name, parameter in parameters.items()
    }
    count = min(samples, len(dataset))
    if not count:
        raise ValueError("Clean source data for Fisher is empty")
    device = next(model.parameters()).device
    for index in range(count):
        logits = model(dataset[index][0][None].to(device))[0]
        loss = F.cross_entropy(logits, logits.detach().argmax(1))
        gradients = torch.autograd.grad(loss, list(parameters.values()))
        for (name, _), gradient in zip(parameters.items(), gradients):
            fisher[name][0].add_(gradient.detach().square(), alpha=1 / count)
    model.requires_grad_(False)
    return fisher


class BaselineAdapter:
    """Use each baseline's state transition while retaining the shared stream API."""

    def __init__(self, model, config, fishers=None):
        self.model, self.config, self.variant = model, config, config["variant"]
        classes = len(model.classifier.weight)
        self.model.eval().requires_grad_(False)
        self.probabilities = None
        self.fishers = fishers
        if self.variant == "t3a":
            self.supports = model.classifier.weight.detach().clone()
            logits = model.classifier(self.supports)
            self.labels = F.one_hot(logits.argmax(1), classes).float()
            self.entropies = entropy(logits).detach()
            self.optimizer = None
        else:
            parameters = configure_bn(model, self.variant == "cotta")
            self.optimizer = torch.optim.Adam(parameters, lr=config["tta_lr"])
            if self.variant == "eata" and fishers is None:
                raise ValueError(
                    "EATA needs clean-source diagonal Fisher; target labels cannot substitute"
                )
            if self.variant == "cotta":
                self.teacher, self.anchor = deepcopy(model), deepcopy(model)
                self.teacher.requires_grad_(False)
                self.anchor.requires_grad_(False)
                self.initial = deepcopy(model.state_dict())
                self.augmentation = transforms.Compose(
                    [
                        transforms.Pad(16, padding_mode="edge"),
                        transforms.ColorJitter(
                            brightness=(0.6, 1.4),
                            contrast=(0.7, 1.3),
                            saturation=(0.5, 1.5),
                            hue=(-0.06, 0.06),
                        ),
                        transforms.RandomAffine(
                            15,
                            translate=(1 / 16, 1 / 16),
                            scale=(0.9, 1.1),
                            interpolation=InterpolationMode.BILINEAR,
                        ),
                        transforms.GaussianBlur(5, sigma=(0.001, 0.5)),
                        transforms.CenterCrop(32),
                        transforms.RandomHorizontalFlip(),
                    ]
                )

    def state_dict(self):
        state = {
            "model": self.model.state_dict(),
            "variant": self.variant,
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "probabilities": self.probabilities,
        }
        if self.variant == "t3a":
            state.update(
                supports=self.supports, labels=self.labels, entropies=self.entropies
            )
        if self.variant == "eata":
            state["fishers"] = self.fishers
        if self.variant == "cotta":
            state.update(
                teacher=self.teacher.state_dict(),
                anchor=self.anchor.state_dict(),
                initial=self.initial,
                rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
            )
        return deepcopy(state)

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        device = next(self.model.parameters()).device
        self.probabilities = (
            state["probabilities"].to(device)
            if state["probabilities"] is not None
            else None
        )
        if self.variant == "t3a":
            self.supports, self.labels, self.entropies = [
                state[key].to(device).clone()
                for key in ("supports", "labels", "entropies")
            ]
        if self.variant == "eata":
            self.fishers = {
                name: [item.to(device) for item in pair]
                for name, pair in state["fishers"].items()
            }
        if self.variant == "cotta":
            self.teacher.load_state_dict(state["teacher"])
            self.anchor.load_state_dict(state["anchor"])
            self.initial = {
                name: value.to(device) for name, value in state["initial"].items()
            }
            torch.set_rng_state(state["rng"])
            if state["cuda_rng"]:
                torch.cuda.set_rng_state_all(state["cuda_rng"])

    def adapt(self, x):
        if self.variant == "t3a":
            with torch.no_grad():
                logits, features = self.model(x)
                self.supports = torch.cat((self.supports, features))
                self.labels = torch.cat(
                    (
                        self.labels,
                        F.one_hot(logits.argmax(1), self.labels.shape[1]).float(),
                    )
                )
                self.entropies = torch.cat((self.entropies, entropy(logits)))
                selected = []
                limit = self.config.get("t3a_filter_k", 20)
                if limit < 1:
                    raise ValueError("t3a_filter_k must be positive")
                for label in range(self.labels.shape[1]):
                    indices = torch.where(self.labels.argmax(1) == label)[0]
                    selected.append(indices[self.entropies[indices].argsort()[:limit]])
                indices = torch.cat(selected)
                self.supports, self.labels, self.entropies = (
                    self.supports[indices],
                    self.labels[indices],
                    self.entropies[indices],
                )
                weights = F.normalize(self.supports, dim=1).T @ self.labels
                predicted = (features @ F.normalize(weights, dim=0)).argmax(1)
            return (
                predicted,
                torch.zeros(len(x), dtype=torch.bool, device=x.device),
                {"loss": 0.0, "threshold": 0.0},
            )
        logits, _ = self.model(x)
        if self.variant == "eata":
            entropies = entropy(logits)
            margin = self.config.get("eata_margin_factor", 0.4) * math.log(
                logits.shape[1]
            )
            mask = entropies.detach() < margin
            if self.probabilities is not None:
                mask &= F.cosine_similarity(
                    logits.detach().softmax(1), self.probabilities[None], dim=1
                ).abs() < self.config.get("eata_diversity", 0.05)
            predicted = logits.detach().argmax(1)
            if not mask.any():
                return predicted, mask, {"loss": 0.0, "threshold": margin}
            current = logits.detach().softmax(1)[mask].mean(0)
            self.probabilities = (
                current
                if self.probabilities is None
                else 0.9 * self.probabilities + 0.1 * current
            )
            loss = (entropies[mask] * (margin - entropies[mask].detach()).exp()).mean()
            for name, parameter in self.model.named_parameters():
                if name in self.fishers:
                    diagonal, reference = self.fishers[name]
                    loss = (
                        loss
                        + self.config.get("eata_fisher_alpha", 2000.0)
                        * (diagonal * (parameter - reference).square()).sum()
                    )
        else:
            with torch.no_grad():
                anchor_confidence = self.anchor(x)[0].softmax(1).max(1).values.mean()
                teacher_logits = self.teacher(x)[0]
                if anchor_confidence < self.config.get("cotta_anchor_threshold", 0.9):
                    count = self.config.get("cotta_augmentations", 32)
                    if count < 1:
                        raise ValueError("cotta_augmentations must be positive")
                    augmented = []
                    for _ in range(count):
                        view = self.augmentation(x.clamp(0, 1))
                        gamma = float(torch.empty(()).uniform_(0.7, 1.3))
                        view = view.clamp(0, 1).pow(gamma) + 0.005 * torch.randn_like(
                            view
                        )
                        augmented.append(self.teacher(view.clamp(0, 1))[0])
                    teacher_logits = torch.stack(augmented).mean(0)
            predicted = teacher_logits.argmax(1)
            mask = torch.ones(len(x), dtype=torch.bool, device=x.device)
            loss = -(teacher_logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.variant == "cotta":
            with torch.no_grad():
                alpha = self.config.get("cotta_teacher_momentum", 0.99)
                restore = self.config.get("cotta_restore", 0.1)
                for teacher, student in zip(
                    self.teacher.parameters(), self.model.parameters()
                ):
                    teacher.mul_(alpha).add_(student, alpha=1 - alpha)
                for name, parameter in self.model.named_parameters():
                    if parameter.requires_grad:
                        reset = torch.rand_like(parameter) < restore
                        parameter.copy_(
                            torch.where(reset, self.initial[name], parameter)
                        )
        return (
            predicted,
            mask.detach(),
            {
                "loss": float(loss.detach()),
                "threshold": margin if self.variant == "eata" else 0.0,
            },
        )
