import numpy as np
from vo.primitives import Features, Frame, Matches
from vo.algorithms import RANSAC
from vo.features import HarrisCornerDetector
import pytest
from scipy.spatial.distance import cdist


@pytest.fixture
def sample_frames():
    # Create dummy data for testing
    descriptors1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    descriptors2 = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])

    features1 = Features(
        keypoints=np.array([[0, 0], [1, 1], [2, 2]]), descriptors=descriptors1
    )
    features2 = Features(
        keypoints=np.array([[0, 0], [1, 1], [2, 2]]), descriptors=descriptors2
    )

    dummy_image1 = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_image2 = np.zeros((100, 100, 3), dtype=np.uint8)

    frame1 = Frame(image=dummy_image1, features=features1)
    frame2 = Frame(image=dummy_image2, features=features2)

    return frame1, frame2


def test_matchDescriptor(sample_frames):
    frame1, frame2 = sample_frames

    # Create an instance of the HarrisCornerDetector class
    matcher = HarrisCornerDetector(match_lambda=4.0)

    # Call the matchDescriptor method
    matches = matcher.matchDescriptor(frame1, frame2)

    # Ensure that the result is of type Matches
    assert isinstance(matches, Matches)

    # Ensure number of keyframes is the same in both frames
    assert (
        matches.frame1.features.keypoints.shape
        == matches.frame2.features.keypoints.shape
    )


def test_matchDescriptor_different_lambda(sample_frames):
    # Test when using a large match_lambda, no matches should be found
    frame1, frame2 = sample_frames

    large_lambda = 100.0
    matcher_l = HarrisCornerDetector(match_lambda=large_lambda)
    matches_l = matcher_l.matchDescriptor(frame1, frame2)

    small_lambda = 0.1
    matcher_s = HarrisCornerDetector(match_lambda=small_lambda)
    matches_s = matcher_s.matchDescriptor(frame1, frame2)

    assert isinstance(matches_l, Matches)
    assert isinstance(matches_s, Matches)
    # Ensure number of keyframes is the same in both frames
    assert (
        matches_l.frame1.features.keypoints.shape
        == matches_l.frame2.features.keypoints.shape
    )
    assert (
        matches_s.frame1.features.keypoints.shape
        == matches_s.frame2.features.keypoints.shape
    )


def test_matchDescriptor_many_keypoints():
    # Test when using a large match_lambda, no matches should be found
    descriptors1 = np.array(
        [[1, 2, 3], [4, 5, 6], [4, 5, 6], [4, 5, 6], [4, 5, 6], [4, 5, 6]]
    )
    descriptors2 = np.array(
        [
            [10, 11, 12],
            [13, 14, 15],
            [10, 11, 12],
            [10, 11, 12],
            [10, 11, 12],
            [10, 11, 12],
        ]
    )

    features1 = Features(
        keypoints=np.array([[0, 0], [1, 1], [3, 3], [45, 40], [0, 50], [20, 80]]),
        descriptors=descriptors1,
    )
    features2 = Features(
        keypoints=np.array([[0, 0], [1, 1], [60, 0], [0, 10], [34, 34], [10, 10]]),
        descriptors=descriptors2,
    )

    dummy_image1 = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_image2 = np.zeros((100, 100, 3), dtype=np.uint8)

    frame1 = Frame(image=dummy_image1, features=features1)
    frame2 = Frame(image=dummy_image2, features=features2)

    # Create an instance of the HarrisCornerDetector class
    matcher = HarrisCornerDetector(match_lambda=4.0)

    # Call the matchDescriptor method
    matches = matcher.matchDescriptor(frame1, frame2)

    # Ensure that the result is of type Matches
    assert isinstance(matches, Matches)

    # Ensure number of keyframes is the same in both frames
    assert (
        matches.frame1.features.keypoints.shape
        == matches.frame2.features.keypoints.shape
    )


# Command: pytest test_harris.py
