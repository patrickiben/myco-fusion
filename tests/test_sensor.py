import unittest
import numpy as np

from myco_fusion.sensor import RealisticNoise, simulate_cross_polarization, simulate_sensor


class SensorTests(unittest.TestCase):
    def test_noise_simulation_bounds_and_quantization(self):
        rng = np.random.default_rng(99)
        clean = np.array([[0.1, 0.5, 0.9]])
        noise_float = RealisticNoise(read_noise_sd=0.001, shot_noise_factor=0.002, gain_error_sd=0.01)
        noisy_float = simulate_sensor(clean, noise_float, rng)
        self.assertEqual(noisy_float.shape, clean.shape)
        self.assertTrue((noisy_float >= 0.0).all() and (noisy_float <= 1.0).all())

        # Quantized 10-bit
        noise_10bit = RealisticNoise(bit_depth=10)
        noisy_10bit = simulate_sensor(clean, noise_10bit, rng)
        # 10-bit values must be integer multiples of 1/1023
        steps = noisy_10bit * 1023
        np.testing.assert_allclose(steps, np.round(steps), atol=1e-8)

    def test_cross_polarization_extinguishes_glare(self):
        diffuse = np.array([[0.1, 0.3, 0.7]])
        flash = np.array([1.0, 1.0, 1.0])
        ks = 0.25

        unpolarized = simulate_cross_polarization(diffuse, ks, flash, cross_polarized=False)
        polarized = simulate_cross_polarization(diffuse, ks, flash, cross_polarized=True)

        # Unpolarized adds glare
        np.testing.assert_allclose(unpolarized, 0.75 * diffuse + 0.25 * flash)
        # Polarized extinguishes glare
        np.testing.assert_allclose(polarized, 0.75 * diffuse)

    def test_invalid_arguments_rejected(self):
        rng = np.random.default_rng(1)
        with self.assertRaises(ValueError):
            simulate_sensor(np.ones((4, 2)), RealisticNoise(), rng)
        with self.assertRaises(ValueError):
            simulate_cross_polarization(np.ones(3), 1.5, np.ones(3))


if __name__ == "__main__":
    unittest.main()
