import cv2
import numpy as np
import pytest

from vo.helpers import to_cartesian_coordinates, to_homogeneous_coordinates
from vo.landmarks.triangulation import LandmarksTriangulator

# Test case highly inspired from Exercise 6 "run_test_eight_point.py"

rng = np.random.default_rng(2023)


# Create fixture for LandmarksTriangulator
@pytest.fixture
def triangulator() -> LandmarksTriangulator:
    return LandmarksTriangulator()


@pytest.fixture
def projection_camera1() -> np.ndarray:
    R = np.eye(3)
    t = np.array([[0, 0, 0]]).T
    M = np.hstack((R, t))
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    return K @ M


@pytest.fixture
def projection_camera2() -> np.ndarray:
    theta = np.pi / 8
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    t = np.array([[1, 1, -1]]).T
    M = np.hstack((R, t))
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])  # same K
    return K @ M


def test_find_fundamental_matrix(
    triangulator: LandmarksTriangulator,
    projection_camera1: np.ndarray,
    projection_camera2: np.ndarray,
    is_normalized: bool = True,
):
    N = 1000

    # Randomly generate 3D points
    landmarks = rng.uniform(-1, 1, size=(N, 3, 1))
    landmarks[:, 2] = landmarks[:, 2] * 5 + 10  # Make z-component positive

    landmarks_hom = to_homogeneous_coordinates(landmarks)

    # Project 3D points into two camera frames
    points1 = projection_camera1.reshape(1, 3, 4) @ landmarks_hom
    points2 = projection_camera2.reshape(1, 3, 4) @ landmarks_hom

    points1 = to_cartesian_coordinates(points1)
    points2 = to_cartesian_coordinates(points2)

    F = triangulator.find_fundamental_matrix(
        points1, points2, is_normalized=is_normalized
    )

    # Check that the fundamental matrix is 3x3
    assert F.shape == (3, 3)

    # Check that the algebraic cost is zero, i.e., x2^T@F@x1 = 0 for all correspondences
    points1_hom = to_homogeneous_coordinates(points1)
    points2_hom = to_homogeneous_coordinates(points2)
    cost_algebraic = np.linalg.norm(
        np.sum(points2_hom.T * (F @ points1_hom.T))
    ) / np.sqrt(N)

    assert cost_algebraic < 1e-10

    # Check with OpenCV implementation
    F_cv, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_8POINT)
    print(F, F_cv)
    assert np.allclose(F, F_cv, atol=1e-4)


# TODO / FIXME: Does not currently work
def test_find_fundamental_normalize(
    triangulator: LandmarksTriangulator,
    projection_camera1: np.ndarray,
    projection_camera2: np.ndarray,
):
    test_find_fundamental_matrix(
        triangulator=triangulator,
        projection_camera1=projection_camera1,
        projection_camera2=projection_camera2,
        is_normalized=False,
    )


if __name__ == "__main__":
    pytest.main(["-v", "-s"])
