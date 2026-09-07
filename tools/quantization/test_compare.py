import unittest

import numpy as np

from compare_http import crop_retention, distribution, summarize


class CropTests(unittest.TestCase):
    def test_full_image(self):
        mask = np.ones((10, 20), dtype=bool)
        self.assertEqual(crop_retention(mask, [0, 0], 2), 1)
        self.assertEqual(crop_retention(mask, [1, 1], 2), 1)

    def test_square_edges_and_center(self):
        mask = np.zeros((10, 20), dtype=bool)
        mask[:, :5] = True
        self.assertEqual(crop_retention(mask, [0, .5], 1), 1)
        self.assertEqual(crop_retention(mask, [1, .5], 1), 0)

    def test_uniform_retention(self):
        self.assertEqual(crop_retention(np.ones((20, 20)), [.5, .5], .5), .5)

    def test_distribution(self):
        result = distribution([0, .02, .04])
        self.assertEqual(result["median"], .02)
        self.assertEqual(result["max"], .04)

    def test_summary_units(self):
        rows = [{"name": "a", "points": {"fp32": [.5, .5], "int8": [.51, .56]},
                 "retention": {"fp32": {"1:1": .9}, "int8": {"1:1": .7}}}]
        result = summarize(rows, "int8")
        self.assertEqual(result["shift_above_5pct"], 1)
        self.assertAlmostEqual(result["max_axis_shift"]["max"], .06)
        self.assertAlmostEqual(result["worst_aspect_crop_retention_loss"]["max"], .2)


if __name__ == "__main__":
    unittest.main()
