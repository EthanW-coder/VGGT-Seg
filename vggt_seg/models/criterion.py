from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn


def dice_loss(logits: Tensor, targets: Tensor) -> Tensor:
    probabilities = logits.sigmoid()
    numerator = 2.0 * (probabilities * targets).sum(dim=-1)
    denominator = probabilities.sum(dim=-1) + targets.sum(dim=-1)
    return 1.0 - (numerator + 1.0) / (denominator + 1.0)


@dataclass
class MatchingWeights:
    classification: float = 0.5
    mask: float = 1.0
    bce: float = 1.0
    dice: float = 1.0


class HungarianMatcher(nn.Module):
    def __init__(self, weights: MatchingWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or MatchingWeights()

    @torch.no_grad()
    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> list[tuple[Tensor, Tensor]]:
        matches = []
        for batch_index, target in enumerate(targets):
            device = outputs["pred_logits"].device
            labels = target["labels"].to(device)
            valid = outputs["point_valid"][batch_index].to(device=device, dtype=torch.bool)
            target_masks = target["masks"].to(device)
            if (
                target_masks.ndim != 2
                or target_masks.shape[0] != labels.numel()
                or target_masks.shape[-1] != valid.numel()
            ):
                raise ValueError("target masks must have shape [K, N] matching predicted points")
            if labels.numel() == 0 or not valid.any():
                empty = torch.empty(0, dtype=torch.long, device=device)
                matches.append((empty, empty))
                continue

            masks = target_masks[:, valid].float()
            logits = outputs["pred_masks"][batch_index, :, valid]
            class_probability = outputs["pred_logits"][batch_index].softmax(dim=-1)
            class_cost = -class_probability[:, labels]
            bce_cost = F.softplus(logits).mean(dim=-1, keepdim=True) - logits @ masks.T / logits.shape[-1]
            probability = logits.sigmoid()
            intersection = probability @ masks.T
            dice_cost = 1.0 - (2.0 * intersection + 1.0) / (
                probability.sum(dim=-1, keepdim=True) + masks.sum(dim=-1).unsqueeze(0) + 1.0
            )
            mask_cost = self.weights.bce * bce_cost + self.weights.dice * dice_cost
            cost = self.weights.classification * class_cost + self.weights.mask * mask_cost
            prediction, ground_truth = linear_sum_assignment(cost.float().cpu().numpy())
            matches.append(
                (
                    torch.as_tensor(prediction, dtype=torch.long, device=device),
                    torch.as_tensor(ground_truth, dtype=torch.long, device=device),
                )
            )
        return matches


class VGGTSegCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        lambda_cls: float = 0.5,
        lambda_mask: float = 1.0,
        lambda_icr: float = 0.5,
        lambda_bce: float = 1.0,
        lambda_dice: float = 1.0,
        no_object_weight: float = 0.1,
        temperature: float = 0.1,
        radius: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_mask = lambda_mask
        self.lambda_icr = lambda_icr
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice
        self.temperature = temperature
        self.radius = radius
        self.matcher = HungarianMatcher(
            MatchingWeights(lambda_cls, lambda_mask, lambda_bce, lambda_dice)
        )
        class_weight = torch.ones(num_classes + 1)
        class_weight[-1] = no_object_weight
        self.register_buffer("class_weight", class_weight)

    def _classification_loss(
        self,
        logits: Tensor,
        targets: list[dict[str, Tensor]],
        matches: list[tuple[Tensor, Tensor]],
    ) -> Tensor:
        labels = torch.full(logits.shape[:2], self.num_classes, dtype=torch.long, device=logits.device)
        for batch_index, (prediction, ground_truth) in enumerate(matches):
            labels[batch_index, prediction] = targets[batch_index]["labels"][ground_truth]
        return F.cross_entropy(logits.transpose(1, 2), labels, weight=self.class_weight.to(logits.device))

    def _mask_loss(
        self,
        logits: Tensor,
        targets: list[dict[str, Tensor]],
        matches: list[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor]:
        bce_losses, dice_losses = [], []
        for batch_index, (prediction, ground_truth) in enumerate(matches):
            if prediction.numel():
                valid = targets[batch_index].get("point_valid")
                if valid is None:
                    valid = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
                predicted = logits[batch_index, prediction][:, valid]
                expected = targets[batch_index]["masks"][ground_truth][:, valid].float()
                bce_losses.append(F.binary_cross_entropy_with_logits(predicted, expected))
                dice_losses.append(dice_loss(predicted, expected).mean())
        if not bce_losses:
            zero = logits.sum() * 0.0
            return zero, zero
        return torch.stack(bce_losses).mean(), torch.stack(dice_losses).mean()

    def _contrastive_loss(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        matches: list[tuple[Tensor, Tensor]],
    ) -> Tensor:
        topology = outputs.get("topology_features")
        if topology is None:
            return outputs["query_features"].sum() * 0.0
        losses = []
        query_count = outputs["query_features"].shape[1]
        all_queries = torch.arange(query_count, device=topology.device)
        for batch_index, (prediction, ground_truth) in enumerate(matches):
            if prediction.numel() == 0:
                continue
            negative_mask = torch.ones(query_count, dtype=torch.bool, device=topology.device)
            negative_mask[prediction] = False
            negatives = all_queries[negative_mask]
            for predicted_index, target_index in zip(prediction, ground_truth):
                anchor = topology[batch_index, target_index]
                positive = outputs["query_features"][batch_index, predicted_index]
                positive_score = torch.exp(
                    F.cosine_similarity(positive, anchor, dim=0) / self.temperature
                )
                if negatives.numel() == 0:
                    losses.append(positive_score * 0.0)
                    continue
                negative_features = outputs["query_features"][batch_index, negatives]
                similarity = torch.exp(
                    F.cosine_similarity(negative_features, anchor.unsqueeze(0), dim=-1) / self.temperature
                )
                center = targets[batch_index]["centers"][target_index]
                distance = (outputs["query_centers"][batch_index, negatives] - center).square().sum(dim=-1)
                spatial_weight = torch.exp(-distance / (2.0 * self.radius**2))
                denominator = positive_score + (spatial_weight * similarity).sum()
                losses.append(-torch.log(positive_score / denominator.clamp_min(1e-8)))
        if not losses:
            return topology.sum() * 0.0
        return torch.stack(losses).mean()

    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        device = outputs["pred_logits"].device
        targets = [
            {
                **{key: value.to(device) for key, value in target.items()},
                "point_valid": outputs["point_valid"][index].to(device=device, dtype=torch.bool),
            }
            for index, target in enumerate(targets)
        ]
        matches = self.matcher(outputs, targets)
        loss_cls = self._classification_loss(outputs["pred_logits"], targets, matches)
        loss_bce, loss_dice = self._mask_loss(outputs["pred_masks"], targets, matches)
        loss_mask = self.lambda_bce * loss_bce + self.lambda_dice * loss_dice
        loss_icr = self._contrastive_loss(outputs, targets, matches)
        total = self.lambda_cls * loss_cls + self.lambda_mask * loss_mask + self.lambda_icr * loss_icr
        return {
            "loss": total,
            "loss_cls": loss_cls.detach(),
            "loss_mask": loss_mask.detach(),
            "loss_bce": loss_bce.detach(),
            "loss_dice": loss_dice.detach(),
            "loss_icr": loss_icr.detach(),
        }
