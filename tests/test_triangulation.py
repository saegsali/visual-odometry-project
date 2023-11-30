import cv2
import numpy as np
import pytest

from vo.helpers import to_cartesian_coordinates, to_homogeneous_coordinates
from vo.landmarks.triangulation import LandmarksTriangulator
from vo.sensors.camera import Camera

# Test case highly inspired from Exercise 6 "run_test_eight_point.py"

rng = np.random.default_rng(2023)


# Create fixture for cameras
@pytest.fixture
def camera1() -> Camera:
    return Camera(intrinsic_matrix=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]))


@pytest.fixture
def camera2(camera1) -> Camera:
    return camera1  # use same camera intrinsics


# Create fixture for LandmarksTriangulator
@pytest.fixture
def triangulator(camera1, camera2) -> LandmarksTriangulator:
    return LandmarksTriangulator(
        camera1=camera1, camera2=camera2, use_ransac=False, use_opencv=False
    )


@pytest.fixture
def triangulator_ransac(camera1, camera2) -> LandmarksTriangulator:
    return LandmarksTriangulator(
        camera1=camera1, camera2=camera2, use_ransac=True, use_opencv=False
    )


@pytest.fixture
def triangulator_opencv(camera1, camera2) -> LandmarksTriangulator:
    return LandmarksTriangulator(
        camera1=camera1, camera2=camera2, use_ransac=False, use_opencv=True
    )


@pytest.fixture
def triangulator_opencv_ransac(camera1, camera2) -> LandmarksTriangulator:
    return LandmarksTriangulator(
        camera1=camera1, camera2=camera2, use_ransac=True, use_opencv=True
    )


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
    use_ransac: bool = False,
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

    if use_ransac:
        F, inliers = triangulator._find_fundamental_matrix_ransac(points1, points2)
    else:
        F = triangulator._find_fundamental_matrix(
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

    if use_ransac:
        assert cost_algebraic < 1e-5
    else:
        assert cost_algebraic < 1e-3

    # Check with OpenCV implementation
    if use_ransac:
        F_cv, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
    else:
        F_cv, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_8POINT)
    F_cv = (
        F_cv * F[-1, -1]
    )  # Rescale to account for normalization that OpenCV does (last element always 1)
    assert np.allclose(F, F_cv, atol=1e-4)


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


def test_find_fundamental_ransac(
    triangulator_ransac: LandmarksTriangulator,
    projection_camera1: np.ndarray,
    projection_camera2: np.ndarray,
):
    test_find_fundamental_matrix(
        triangulator=triangulator_ransac,
        projection_camera1=projection_camera1,
        projection_camera2=projection_camera2,
        use_ransac=True,
    )


def test_find_fundamental_opencv(
    triangulator_opencv: LandmarksTriangulator,
    projection_camera1: np.ndarray,
    projection_camera2: np.ndarray,
):
    test_find_fundamental_matrix(
        triangulator=triangulator_opencv,
        projection_camera1=projection_camera1,
        projection_camera2=projection_camera2,
        use_ransac=False,
    )


def test_find_fundamental_opencv_ransac(
    triangulator_opencv_ransac: LandmarksTriangulator,
    projection_camera1: np.ndarray,
    projection_camera2: np.ndarray,
):
    test_find_fundamental_matrix(
        triangulator=triangulator_opencv_ransac,
        projection_camera1=projection_camera1,
        projection_camera2=projection_camera2,
        use_ransac=True,
    )


if __name__ == "__main__":
    pytest.main(["-v", "-s"])
