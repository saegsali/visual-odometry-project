import numpy as np
import pytest
from vo.primitives.features import Features


def test_features_initialization():
    keypoints = np.array([[1, 2], [3, 4]]).reshape(-1, 2, 1)
    landmarks = np.array([[9, 10, 11], [12, 13, 14]]).reshape(-1, 3, 1)

    features = Features(keypoints, landmarks)

    assert np.array_equal(features.keypoints, keypoints)
    assert np.array_equal(features.landmarks, landmarks)


def test_features_default_initialization():
    keypoints = np.array([[1, 2], [3, 4]]).reshape(-1, 2, 1)

    features = Features(keypoints)

    assert np.array_equal(features.keypoints, keypoints)
    assert features.descriptors is None
    assert np.all(np.isnan(features.landmarks))


def test_features_properties():
    keypoints = np.array([[1, 2], [3, 4]]).reshape(-1, 2, 1)

    features = Features(keypoints)

    assert features.length == 2
    assert np.array_equal(features.keypoints, keypoints)
    assert np.allclose(features.state, 0)


if __name__ == "__main__":
    pytest.main()
