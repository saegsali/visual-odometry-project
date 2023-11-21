import unittest
import numpy as np
from vo.primitives.features import Features


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.keypoints = np.array([[1, 2], [3, 4], [5, 6]])
        self.descriptors = np.array([[7, 8], [9, 10], [11, 12]])
        self.landmarks = np.array([[13, 14, 15], [16, 17, 18], [19, 20, 21]])
        self.features = Features(self.keypoints, self.descriptors, self.landmarks)

    def test_init(self):
        self.assertTrue(np.array_equal(self.features.keypoints, self.keypoints))
        self.assertTrue(np.array_equal(self.features.descriptors, self.descriptors))
        self.assertTrue(np.array_equal(self.features.landmarks, self.landmarks))
        self.assertTrue(np.array_equal(self.features.inliers, np.ones(3, dtype=bool)))

    def test_set_descriptors(self):
        new_descriptors = np.array([[22, 23], [24, 25], [26, 27]])
        with self.assertRaises(AssertionError):
            self.features.descriptors = new_descriptors

    def test_set_landmarks(self):
        new_landmarks = np.array([[28, 29, 30], [31, 32, 33], [34, 35, 36]])
        with self.assertRaises(AssertionError):
            self.features.landmarks = new_landmarks

    def test_set_inliers(self):
        new_inliers = np.array([True, False, True])
        self.features.inliers = new_inliers
        self.assertTrue(np.array_equal(self.features.inliers, new_inliers))

    def test_length(self):
        self.assertEqual(self.features.length, 3)

    def test_apply_inliers(self):
        new_inliers = np.array([True, False, True])
        self.features.apply_inliers(new_inliers)
        self.assertTrue(
            np.array_equal(self.features.keypoints, self.keypoints[new_inliers])
        )
        self.assertTrue(
            np.array_equal(self.features.descriptors, self.descriptors[new_inliers])
        )
        self.assertTrue(
            np.array_equal(self.features.landmarks, self.landmarks[new_inliers])
        )


if __name__ == "__main__":
    unittest.main()
