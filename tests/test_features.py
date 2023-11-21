import numpy as np
import pytest
from vo.primitives.features import Features


def test_features_initialization():
    keypoints = np.array([[1, 2], [3, 4]])
    descriptors = np.array([[5, 6], [7, 8]])
    landmarks = np.array([[9, 10, 11], [12, 13, 14]])

    features = Features(keypoints, descriptors, landmarks)

    assert np.array_equal(features.keypoints, keypoints)
    assert np.array_equal(features.descriptors, descriptors)
    assert np.array_equal(features.landmarks, landmarks)


def test_features_default_initialization():
    keypoints = np.array([[1, 2], [3, 4]])

    features = Features(keypoints)

    assert np.array_equal(features.keypoints, keypoints)
    assert features.descriptors is None
    assert features.landmarks is None


def test_features_properties():
    keypoints = np.array([[1, 2], [3, 4]])
    descriptors = np.array([[5, 6], [7, 8]])
    landmarks = np.array([[9, 10, 11], [12, 13, 14]])

    features = Features(keypoints, descriptors, landmarks)

    assert features.length == 2
    assert np.array_equal(features.keypoints, keypoints)
    assert np.array_equal(features.descriptors, descriptors)
    assert np.array_equal(features.landmarks, landmarks)
    assert np.array_equal(
        features.inliers, np.ones(shape=(features.length)).astype(bool)
    )


def test_descriptors_setter():
    keypoints = np.array([[1, 2], [3, 4]])
    descriptors = np.array([[5, 6], [7, 8], [9, 10]])  # wrong number of descriptors

    features = Features(keypoints)

    with pytest.raises(AssertionError):
        features.descriptors = descriptors

    descriptors = np.array([[9, 10], [11, 12]])

    features.descriptors = descriptors

    with pytest.raises(AssertionError):
        features.descriptors = descriptors  # descriptors already set

    assert np.array_equal(features.descriptors, descriptors)


def test_landmarks_setter():
    keypoints = np.array([[1, 2], [3, 4]])
    landmarks = np.array(
        [[5, 6, 7], [8, 9, 10], [11, 12, 13]]
    )  # wrong number of landmarks

    features = Features(keypoints)

    with pytest.raises(AssertionError):
        features.landmarks = landmarks

    landmarks = np.array([[9, 10, 11], [12, 13, 14]])

    features.landmarks = landmarks

    with pytest.raises(AssertionError):
        features.landmarks = landmarks  # landmarks already set

    assert np.array_equal(features.landmarks, landmarks)


if __name__ == "__main__":
    pytest.main()
