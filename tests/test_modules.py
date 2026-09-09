import unittest

import torch
from torch import nn

from vggt_seg.config import ModelConfig
from vggt_seg.models.backbone import EncoderOutput
from vggt_seg.models.criterion import VGGTSegCriterion
from vggt_seg.models.decoder import TopologyRectifiedInterleavedDecoder
from vggt_seg.models.model import VGGTSeg
from vggt_seg.models.sampling import attention_guided_fps, farthest_point_sampling


class SamplingTest(unittest.TestCase):
    def test_attention_guided_fps_starts_at_semantic_maximum(self):
        points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
        attention = torch.tensor([[0.0, 5.0, 1.0]])
        indices = attention_guided_fps(points, attention, 2)
        self.assertEqual(indices[0, 0].item(), 1)
        self.assertEqual(indices.unique().numel(), 2)

    def test_fps_respects_valid_points(self):
        points = torch.randn(1, 8, 3)
        valid = torch.tensor([[True, True, True, False, False, False, False, False]])
        indices = farthest_point_sampling(points, 3, valid)
        self.assertTrue((indices < 3).all())

    def test_invalid_nan_points_are_ignored(self):
        points = torch.tensor([[[0.0, 0.0, 0.0], [float("nan"), 1.0, 0.0], [2.0, 0.0, 0.0]]])
        valid = torch.tensor([[True, False, True]])
        indices = attention_guided_fps(points, torch.tensor([[1.0, 2.0, 3.0]]), 2, valid=valid)
        self.assertTrue(torch.isfinite(indices.float()).all())
        self.assertTrue((indices != 1).all())


class DecoderTest(unittest.TestCase):
    def test_eval_does_not_generate_topology_queries(self):
        decoder = TopologyRectifiedInterleavedDecoder(24, 4, 2, 5, 0.5, 0.15).eval()

        def fail(*_args, **_kwargs):
            raise AssertionError("TQR must not run during inference")

        decoder.topology.forward = fail
        queries = torch.randn(1, 6, 24)
        memory = torch.randn(1, 10, 24)
        points = torch.randn(1, 10, 3)
        valid = torch.ones(1, 10, dtype=torch.bool)
        centers = torch.randn(1, 2, 3)
        covariance = torch.eye(3).view(1, 1, 3, 3).expand(1, 2, 3, 3)
        output = decoder(queries, memory, memory, points, valid, centers, covariance, valid[:, :2])
        self.assertIsNone(output.topology_features)
        self.assertEqual(output.mask_logits.shape, (1, 6, 10))


class CriterionTest(unittest.TestCase):
    def test_full_loss_is_finite(self):
        outputs = {
            "pred_logits": torch.randn(1, 4, 4, requires_grad=True),
            "pred_masks": torch.randn(1, 4, 6, requires_grad=True),
            "query_features": torch.randn(1, 4, 8, requires_grad=True),
            "query_centers": torch.randn(1, 4, 3),
            "topology_features": torch.randn(1, 2, 8, requires_grad=True),
            "point_valid": torch.tensor([[True, True, True, True, False, False]]),
        }
        targets = [
            {
                "labels": torch.tensor([0, 2]),
                "masks": torch.tensor(
                    [[True, True, False, False, False, False], [False, False, True, True, False, False]]
                ),
                "centers": torch.randn(2, 3),
                "covariance": torch.eye(3).repeat(2, 1, 1),
                "d2_to_d1": torch.eye(4),
            }
        ]
        loss = VGGTSegCriterion(3)(outputs, targets)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


class FakeEncoder(nn.Module):
    def forward(self, images, frame_valid=None):
        batch = images.shape[0]
        point_count = 12
        points = torch.randn(batch, point_count, 3, device=images.device)
        return EncoderOutput(
            points=points,
            colors=torch.rand(batch, point_count, 3, device=images.device),
            confidence=torch.ones(batch, point_count, device=images.device),
            features=[torch.randn(batch, point_count, 2048, device=images.device) for _ in range(4)],
            shallow_attention=torch.randn(batch, 2, point_count, device=images.device),
            point_valid=torch.ones(batch, point_count, dtype=torch.bool, device=images.device),
        )


class ModelTest(unittest.TestCase):
    def test_full_train_and_inference_paths(self):
        config = ModelConfig(
            hidden_dim=24,
            num_heads=4,
            decoder_layers=2,
            semantic_queries=4,
            learnable_queries=2,
        )
        model = VGGTSeg(config, encoder=FakeEncoder())
        images = torch.rand(1, 2, 3, 14, 14)
        topology = {
            "centers": torch.randn(1, 2, 3),
            "covariance": torch.eye(3).repeat(1, 2, 1, 1),
            "valid": torch.ones(1, 2, dtype=torch.bool),
        }
        training_output = model.train()(images, topology_targets=topology)
        self.assertEqual(training_output["pred_masks"].shape, (1, 6, 12))
        self.assertIsNotNone(training_output["topology_features"])
        inference_output = model.eval()(images, topology_targets=topology)
        self.assertIsNone(inference_output["topology_features"])

    def test_vanilla_uses_only_sampled_queries(self):
        config = ModelConfig(
            variant="vanilla",
            hidden_dim=24,
            num_heads=4,
            decoder_layers=1,
            semantic_queries=4,
            learnable_queries=2,
        )
        output = VGGTSeg(config, encoder=FakeEncoder())(torch.rand(1, 2, 3, 14, 14))
        self.assertEqual(output["pred_masks"].shape[1], 6)
        self.assertIsNone(output["topology_features"])


if __name__ == "__main__":
    unittest.main()
