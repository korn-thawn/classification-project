"""Unit tests for vision.preprocess."""
import copy
import unittest
from pathlib import Path

import numpy as np

from vision.config import load_config
from vision.preprocess import crop_roi, normalize, prepare_uint8, preprocess

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class TestPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = copy.deepcopy(load_config(CONFIG_PATH))
        self.cfg["preprocess"]["roi"] = None
        rng = np.random.default_rng(0)
        self.img = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)

    def test_output_shape_and_dtype(self) -> None:
        out = preprocess(self.img, self.cfg)
        size = self.cfg["preprocess"]["image_size"]
        self.assertEqual(out.shape, (3, size, size))
        self.assertEqual(out.dtype, np.float32)

    def test_crop_roi_coordinates(self) -> None:
        crop = crop_roi(self.img, [10, 20, 100, 50])
        self.assertEqual(crop.shape, (50, 100, 3))
        np.testing.assert_array_equal(crop, self.img[20:70, 10:110])

    def test_crop_roi_none_is_identity(self) -> None:
        self.assertIs(crop_roi(self.img, None), self.img)

    def test_roi_outside_image_raises(self) -> None:
        with self.assertRaises(ValueError):
            crop_roi(self.img, [600, 0, 100, 100])
        with self.assertRaises(ValueError):
            crop_roi(self.img, [-1, 0, 10, 10])

    def test_roi_applied_in_prepare(self) -> None:
        self.cfg["preprocess"]["roi"] = [0, 0, 200, 100]
        size = self.cfg["preprocess"]["image_size"]
        self.assertEqual(prepare_uint8(self.img, self.cfg).shape, (size, size, 3))

    def test_normalize_values_and_channel_order(self) -> None:
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        img[..., 0] = 255  # blue channel in BGR
        out = normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # RGB order after normalize: R=-1, G=-1, B=+1
        np.testing.assert_allclose(out[0], -1.0)
        np.testing.assert_allclose(out[1], -1.0)
        np.testing.assert_allclose(out[2], 1.0)

    def test_grayscale_and_bgra_inputs(self) -> None:
        size = self.cfg["preprocess"]["image_size"]
        gray = self.img[..., 0]
        bgra = np.dstack([self.img, np.full(self.img.shape[:2], 255, np.uint8)])
        self.assertEqual(preprocess(gray, self.cfg).shape, (3, size, size))
        self.assertEqual(preprocess(bgra, self.cfg).shape, (3, size, size))

    def test_train_path_matches_inference_path(self) -> None:
        p = self.cfg["preprocess"]
        train_path = normalize(prepare_uint8(self.img, self.cfg), p["mean"], p["std"])
        np.testing.assert_array_equal(train_path, preprocess(self.img, self.cfg))


if __name__ == "__main__":
    unittest.main()
