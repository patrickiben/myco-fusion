import unittest
import numpy as np
from myco_fusion.geometry import project_spectra, combine_views


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.cube = np.ones((4, 4, 27)) * 0.6

    def test_visibility_missing_and_image_boundary(self):
        p = np.array([[0, 0, 1], [0, 0, 2], [0, 0, -1], [0, 0, 0], [3.75, 0, 1], [3.49, 1, 1]])
        out = project_spectra(p, self.cube, np.eye(3), np.eye(4))
        np.testing.assert_array_equal(out["n_views"], [1, 0, 0, 0, 0, 1])
        self.assertTrue(np.isnan(out["spectra"][1:5]).all())
        np.testing.assert_allclose(out["spectra"][[0, 5]], 0.6)

    def test_world_to_camera_direction(self):
        T = np.eye(4)
        T[:3, 3] = [-10, 0, 1]
        result = project_spectra([[10, 0, 0]], self.cube, np.eye(3), T)
        self.assertEqual(result["n_views"][0], 1)

    def test_nonfinite_inputs_and_cube_mask(self):
        with self.assertRaises(ValueError):
            project_spectra([[np.nan, 0, 1]], self.cube, np.eye(3), np.eye(4))
        self.cube[0, 0, 5] = np.nan
        out = project_spectra([[0, 0, 1]], self.cube, np.eye(3), np.eye(4))
        self.assertEqual(out["n_views"][0], 0)

    def test_empty_geometry_and_multiview(self):
        out = project_spectra(np.empty((0, 3)), self.cube, np.eye(3), np.eye(4))
        self.assertEqual(out["spectra"].shape, (0, 27))
        a = {"spectra": np.array([[.2, .4], [np.nan, np.nan]]), "n_views": np.array([1, 0])}
        b = {"spectra": np.array([[.6, .8], [np.nan, np.nan]]), "n_views": np.array([1, 0])}
        out = combine_views([a, b])
        np.testing.assert_allclose(out["spectra"][0], [.4, .6])
        self.assertTrue(np.isnan(out["spectra"][1]).all())
        np.testing.assert_array_equal(out["n_views"], [2, 0])

    def test_reject_nonrigid_transform(self):
        T = np.eye(4); T[0, 0] = 2
        with self.assertRaises(ValueError):
            project_spectra([[0, 0, 1]], self.cube, np.eye(3), T)
