import cv2
import numpy as np
import pytest

from vo.helpers import to_homogeneous_coordinates
from vo.landmarks.triangulation import LandmarksTriangulator
from vo.sensors.camera import Camera

# Test case highly inspired from Exercise 6 "run_test_eight_point.py"

rng = np.random.default_rng(2023)


# Create fixture for cameras
@pytest.fixture
def camera1() -> Camera:
    return Camera(
        intrinsic_matrix=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
        R=np.eye(3),
        t=np.array([[0, 0, 0]]).T,
    )


@pytest.fixture
def camera2() -> Camera:
    theta = np.pi / 8
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    t = np.array([[1, 1, -1]]).T
    return Camera(
        intrinsic_matrix=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]), R=R, t=t
    )


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


def test_find_fundamental_matrix(
    triangulator: LandmarksTriangulator,
    camera1: Camera,
    camera2: Camera,
    is_normalized: bool = True,
    use_ransac: bool = False,
):
    N = 1000

    # Randomly generate 3D points
    landmarks = rng.uniform(-1, 1, size=(N, 3, 1))
    landmarks[:, 2] = landmarks[:, 2] * 5 + 10  # Make z-component positive

    # Project 3D points into two camera frames
    points1 = camera1.project_points_world_frame(landmarks)
    points2 = camera2.project_points_world_frame(landmarks)

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
    camera1: Camera,
    camera2: Camera,
):
    test_find_fundamental_matrix(
        triangulator=triangulator,
        camera1=camera1,
        camera2=camera2,
        is_normalized=False,
    )


def test_find_fundamental_ransac(
    triangulator_ransac: LandmarksTriangulator,
    camera1: Camera,
    camera2: Camera,
):
    test_find_fundamental_matrix(
        triangulator=triangulator_ransac,
        camera1=camera1,
        camera2=camera2,
        use_ransac=True,
    )


def test_find_fundamental_opencv(
    triangulator_opencv: LandmarksTriangulator,
    camera1: Camera,
    camera2: Camera,
):
    test_find_fundamental_matrix(
        triangulator=triangulator_opencv,
        camera1=camera1,
        camera2=camera2,
        use_ransac=False,
    )


def test_find_fundamental_opencv_ransac(
    triangulator_opencv_ransac: LandmarksTriangulator,
    camera1: Camera,
    camera2: Camera,
):
    test_find_fundamental_matrix(
        triangulator=triangulator_opencv_ransac,
        camera1=camera1,
        camera2=camera2,
        use_ransac=True,
    )


def test_linear_triangulation(
    triangulator: LandmarksTriangulator,
    camera1: Camera,
    camera2: Camera,
):
    N = 1000

    # Randomly generate 3D points
    landmarks = rng.uniform(-1, 1, size=(N, 3, 1))
    landmarks[:, 2] = landmarks[:, 2] * 5 + 10  # Make z-component positive

    # Project 3D points into two camera frames
    points1 = camera1.project_points_world_frame(landmarks)
    points2 = camera2.project_points_world_frame(landmarks)

    # Linear triangulation
    M2 = triangulator._find_relative_pose(points1, points2)

    assert np.allclose(M2[:3, :3], camera2.R), "Rotation matrix is not correct"
    assert np.allclose(
        M2[:3, 3:] / np.linalg.norm(M2[:3, 3:]), camera2.t / np.linalg.norm(camera2.t)
    ), "Translation vector is not correct (up to scale)"

    # Find correct scale of translation vector
    scale = np.linalg.norm(camera2.t) / np.linalg.norm(M2[:3, 3:])
    M2[:3, 3:] = M2[:3, 3:] * scale

    # Triangulate landmarks (with groundtruth scale)
    c2_T_c1 = np.vstack([M2, [0, 0, 0, 1]])
    c2_T_w = c2_T_c1 @ camera1.c_T_w

    landmarks_triangulated = triangulator._linear_triangulation(
        points1,
        points2,
        C1=camera1.intrinsic_matrix @ camera1.c_T_w[:3],
        C2=camera2.intrinsic_matrix @ c2_T_w[:3],
    )

    # Check that the triangulated landmarks are close to the original landmarks
    assert np.allclose(landmarks, landmarks_triangulated, atol=1e-4)


if __name__ == "__main__":
    pytest.main(["-v", "-s"])
