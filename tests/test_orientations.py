import unittest

import numpy as np

from ligandparam.io.orientations import (
    N_ORIENTATIONS_SO3_N28,
    get_quaternion_pack,
    legacy_euler_kwargs,
    minimum_pairwise_rotation_angle,
    quaternion_to_matrix,
)


class TestQuaternionOrientations(unittest.TestCase):
    def test_so3_n28_pack_is_normalized_and_well_separated(self):
        quaternions = get_quaternion_pack("so3_n28")

        self.assertEqual(quaternions.shape, (N_ORIENTATIONS_SO3_N28, 4))
        self.assertTrue(np.allclose(np.linalg.norm(quaternions, axis=1), 1.0))
        self.assertTrue(np.allclose(quaternions[0], [1.0, 0.0, 0.0, 0.0]))
        self.assertGreater(minimum_pairwise_rotation_angle(quaternions), 60.0)

    def test_quaternion_matrices_are_proper_rotations(self):
        for quaternion in get_quaternion_pack("so3_n28"):
            rotation = quaternion_to_matrix(quaternion)

            self.assertTrue(
                np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            )
            self.assertTrue(np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12))

    def test_legacy_euler_kwargs_match_orientation_count(self):
        kwargs = legacy_euler_kwargs()
        n = len(kwargs["alpha"]) * len(kwargs["beta"]) * len(kwargs["gamma"])
        self.assertEqual(kwargs["orientation_protocol"], "legacy_euler")
        self.assertEqual(n, N_ORIENTATIONS_SO3_N28)


if __name__ == "__main__":
    unittest.main()
