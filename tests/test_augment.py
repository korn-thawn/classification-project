"""Unit tests for vision.augment."""
import copy
import unittest
from pathlib import Path

import numpy as np

from vision import augment as A
from vision.config import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def all_on_config() -> dict:
    """Augment config with every op enabled and p=1."""
    cfg = copy.deepcopy(load_config(CONFIG_PATH)["augment"])
    cfg["enabled"] = True
    for name in A.Augmenter.ORDER:
        cfg[name]["enabled"] = True
        cfg[name]["p"] = 1.0
    return cfg


class TestAugmentOps(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.img = rng.integers(30, 220, size=(64, 80, 3), dtype=np.uint8)

    def test_ops_keep_shape_and_dtype(self) -> None:
        rng = np.random.default_rng(1)
        outs = [
            A.hflip(self.img), A.vflip(self.img),
            A.affine(self.img, 10, 3, -2, 1.05),
            A.brightness(self.img, 20), A.contrast(self.img, 1.2), A.gamma(self.img, 0.8),
            A.gaussian_noise(self.img, 5.0, rng), A.gaussian_blur(self.img, 3),
        ]
        for out in outs:
            self.assertEqual(out.shape, self.img.shape)
            self.assertEqual(out.dtype, np.uint8)

    def test_flip_direction(self) -> None:
        np.testing.assert_array_equal(A.hflip(self.img)[:, 0], self.img[:, -1])
        np.testing.assert_array_equal(A.vflip(self.img)[0], self.img[-1])

    def test_brightness_direction_and_clip(self) -> None:
        self.assertGreater(A.brightness(self.img, 20).mean(), self.img.mean())
        self.assertLess(A.brightness(self.img, -20).mean(), self.img.mean())
        self.assertEqual(A.brightness(self.img, 1000).min(), 255)

    def test_contrast_changes_spread(self) -> None:
        self.assertGreater(A.contrast(self.img, 1.5).std(), self.img.std())
        self.assertLess(A.contrast(self.img, 0.5).std(), self.img.std())

    def test_gamma_one_is_identity(self) -> None:
        np.testing.assert_array_equal(A.gamma(self.img, 1.0), self.img)
        self.assertGreater(A.gamma(self.img, 0.5).mean(), self.img.mean())

    def test_identity_parameters(self) -> None:
        rng = np.random.default_rng(0)
        np.testing.assert_array_equal(A.gaussian_noise(self.img, 0.0, rng), self.img)
        np.testing.assert_array_equal(A.affine(self.img, 0, 0, 0, 1.0), self.img)
        np.testing.assert_array_equal(A.gaussian_blur(self.img, 1), self.img)

    def test_blur_rejects_even_kernel(self) -> None:
        with self.assertRaises(ValueError):
            A.gaussian_blur(self.img, 4)


class TestAugmenter(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.img = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)

    def test_master_switch_off_is_identity(self) -> None:
        cfg = all_on_config()
        cfg["enabled"] = False
        np.testing.assert_array_equal(A.Augmenter(cfg, seed=0)(self.img), self.img)

    def test_all_ops_disabled_is_identity(self) -> None:
        cfg = all_on_config()
        for name in A.Augmenter.ORDER:
            cfg[name]["enabled"] = False
        np.testing.assert_array_equal(A.Augmenter(cfg, seed=0)(self.img), self.img)

    def test_same_seed_same_result(self) -> None:
        cfg = all_on_config()
        out1 = A.Augmenter(cfg, seed=123)(self.img)
        out2 = A.Augmenter(cfg, seed=123)(self.img)
        np.testing.assert_array_equal(out1, out2)
        self.assertFalse(np.array_equal(out1, A.Augmenter(cfg, seed=124)(self.img)))

    def test_all_on_keeps_shape_and_dtype(self) -> None:
        out = A.Augmenter(all_on_config(), seed=0)(self.img)
        self.assertEqual(out.shape, self.img.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_input_not_modified(self) -> None:
        before = self.img.copy()
        A.Augmenter(all_on_config(), seed=0)(self.img)
        np.testing.assert_array_equal(self.img, before)


if __name__ == "__main__":
    unittest.main()
