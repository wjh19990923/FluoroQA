import gc
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from fluoroqa.calibration import fit_threshold, evaluate, local_score
from fluoroqa.engine import save_checkpoint, load_checkpoint
from fluoroqa.geometry import current_neighbors
from fluoroqa.losses import BCEPolicy, objective
from fluoroqa.preprocess import fluoroscopy, absorption_drr


class CoreTests(unittest.TestCase):
    def test_margins(self):
        dp, dn = torch.tensor([1., 1., 1.]), torch.tensor([.9, .9, .9])
        result = objective(dp, dn, [0., 2., 20.])
        self.assertAlmostEqual(float(result["loss"]), (.1+.102+.11)/3, places=6)
        self.assertAlmostEqual(float(objective(dp, dn, [0, 2, 20], margin="zero")["loss"]), .1, places=6)

    def test_bce_boundary_and_gradient_direction(self):
        dp = torch.tensor([.5, .5], requires_grad=True)
        dn = torch.tensor([.5, .5], requires_grad=True)
        result = objective(dp, dn, [1.999, 2.], bce=BCEPolicy(.5, .1, 1))
        result["bce"].backward()
        self.assertTrue((dp.grad > 0).all())
        self.assertGreater(dn.grad[0], 0)
        self.assertLess(dn.grad[1], 0)
        self.assertAlmostEqual(float(result["bce"]), np.log(2), places=6)

    def test_invalid_losses(self):
        with self.assertRaises(ValueError):
            objective(torch.ones(2), torch.ones(1), [1, 2])
        with self.assertRaises(ValueError):
            objective(torch.ones(1), torch.ones(1), [-1])
        with self.assertRaises(ValueError):
            BCEPolicy(0, 0, 1)

    def test_threshold_ties_and_auc(self):
        threshold = fit_threshold([.1, .2, .8, .9], [0, 1, 2, 3], split="train")
        self.assertEqual(threshold, .8)
        self.assertEqual(evaluate([.1, .2, .8, .9], [0, 1, 2, 3], threshold)["auroc"], 1)
        tie = fit_threshold([.5, .5], [0, 2], split="train")
        self.assertEqual(tie, np.nextafter(.5, np.inf))
        self.assertEqual(evaluate([.5, .5], [0, 2], tie)["auroc"], .5)
        self.assertIsNone(evaluate([.1], [0], .5)["auroc"])

    def test_threshold_split_guards(self):
        for split in ("val", "test", "heldout"):
            with self.assertRaises(ValueError):
                fit_threshold([0, 1], [0, 2], split=split)
        fit_threshold([0, 1], [0, 2], split="val", mode="local")

    def test_local(self):
        neighbors = np.arange(24).reshape(2, 12)
        center = np.array([3., 20.])
        np.testing.assert_allclose(local_score(center, neighbors), center-neighbors.mean(1))
        np.testing.assert_allclose(local_score(center, neighbors, "z"), (center-neighbors.mean(1))/neighbors.std(1))
        np.testing.assert_allclose(local_score([2.], np.ones((1, 12)), "z"), [1.])

    def test_current_pose_geometry(self):
        pose = np.eye(4); pose[:3, 3] = [8, -3, 11]
        center = np.array([1., 2., 3.]); pivot = pose[:3, :3]@center+pose[:3, 3]
        neighbors = current_neighbors(pose, center)
        self.assertEqual(neighbors.shape, (12, 4, 4))
        np.testing.assert_allclose(neighbors[0][:3, 3], [6, -3, 11])
        for neighbor in neighbors[6:]:
            np.testing.assert_allclose(neighbor[:3, :3]@center+neighbor[:3, 3], pivot)
        np.testing.assert_allclose(neighbors[6][:3, :3].T, neighbors[7][:3, :3], atol=1e-14)

    def test_preprocessing(self):
        fixed = fluoroscopy(np.ones((120, 120)), 32)
        self.assertEqual(tuple(fixed.shape), (1, 1, 32, 32))
        torch.testing.assert_close(fixed, torch.ones_like(fixed))
        moving = absorption_drr(np.array([[0., 1.], [2., 3.]]))
        self.assertEqual(float(moving[0, 0, 0, 0]), 1)
        self.assertEqual(float(moving[0, 0, 1, 1]), 0)
        with self.assertRaises(ValueError):
            absorption_drr(np.ones((32, 32)))

    def test_checkpoint(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters())
        model(torch.ones(1, 2)).sum().backward(); optimizer.step()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"test.pt"
            save_checkpoint(path, model, optimizer, 1, {"seed": 1}, "synthetic")
            other = torch.nn.Linear(2, 1)
            opt = torch.optim.Adam(other.parameters())
            payload = load_checkpoint(path, other, opt)
            self.assertEqual(payload["epoch"], 1)
            for a, b in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(a, b)
                self.assertEqual(float(opt.state[b]["step"]), 1)
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, model, optimizer, 2, {}, "synthetic")

    def test_real_vgg_freeze_and_forward_offline(self):
        from fluoroqa.model import FluoroQA
        torch.set_num_threads(2)
        model = FluoroQA(pretrained=False)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 1472)
        model.eval()
        x = torch.rand(1, 1, 32, 32)
        with torch.no_grad():
            self.assertAlmostEqual(float(model(x, x)), 0., places=7)
        model.set_backbone("tail4")
        convs = [m for m in model.metric.net.modules() if isinstance(m, torch.nn.Conv2d)]
        self.assertTrue(all(not m.weight.requires_grad for m in convs[:9]))
        self.assertTrue(all(m.weight.requires_grad for m in convs[9:]))
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 9440704)
        model.train()
        self.assertFalse(model.metric.net.training)
        self.assertTrue(model.metric.lins.training)
        loss = model(x, torch.rand_like(x)).sum()
        loss.backward()
        self.assertTrue(all(m.weight.grad is None for m in convs[:9]))
        self.assertTrue(all(m.weight.grad is not None and torch.isfinite(m.weight.grad).all() for m in convs[9:]))
        del model; gc.collect()


if __name__ == "__main__":
    unittest.main()
