import cv2
import numpy as np
import pytest
from vo.primitives import Features
from vo.sensors import Camera
from vo.pose_estimation import P3PPoseEstimator

rng = np.random.default_rng(2023)


# Create fixture for cameras
@pytest.fixture
def camera1() -> Camera:
    return Camera(
        intrinsic_matrix=np.array(
            [[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float
        ),
        R=np.eye(3),
        t=np.array([[0, 0, 0]]).T,
    )


@pytest.fixture
def camera2() -> Camera:
    theta1 = np.pi / 8
    theta2 = np.pi / 32
    R = np.array(
        [
            [np.cos(theta1), -np.sin(theta1), 0],
            [np.sin(theta1), np.cos(theta1), 0],
            [0, 0, 1],
        ]
    )
    R = R @ np.array(
        [
            [np.cos(theta2), 0, np.sin(theta2)],
            [0, 1, 0],
            [-np.sin(theta2), 0, np.cos(theta2)],
        ]
    )
    t = np.array([[1, 1, -1]]).T
    return Camera(
        intrinsic_matrix=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]), R=R, t=t
    )


# Create fixture for P3PPoseEstimator
@pytest.fixture
def pose_estimator(camera1) -> P3PPoseEstimator:
    return P3PPoseEstimator(
        intrinsic_matrix=camera1.intrinsic_matrix,
        inlier_threshold=1,
        outlier_ratio=0.9,
        confidence=0.99,
        max_iterations=1000,
    )


# Create test for P3PPoseEstimator
def test_estimate_pose(
    pose_estimator: P3PPoseEstimator, camera1: Camera, camera2: Camera
):
    N = 1000

    # Randomly generate 3D points
    landmarks = rng.uniform(-1, 1, size=(N, 3, 1))
    landmarks[:, 2] = landmarks[:, 2] * 5 + 10  # Make z-component positive

    # Project 3D points into two camera frames
    points1 = camera1.project_points_world_frame(landmarks)
    points2 = camera2.project_points_world_frame(landmarks)

    assert (
        len(landmarks) == len(points1) == len(points2)
    ), "Number of points must be equal"

    # Create features
    features2 = Features(keypoints=points2, landmarks=landmarks)

    # Estimate pose
    model, inliers = pose_estimator.estimate_pose(features2)
    rmatrix, tvec = model

    assert rmatrix.shape == (3, 3), "Rotation matrix must be 3x3"
    assert tvec.shape == (3, 1), "Translation vector must be 3x1"

    # Estimate pose with opencv
    success, rvec_cv, tvec_cv, inliers_cv = cv2.solvePnPRansac(
        landmarks, points2, camera1.intrinsic_matrix, None, flags=cv2.SOLVEPNP_P3P
    )
    assert success, "OpenCV P3P failed"
    rmatrix_cv = cv2.Rodrigues(rvec_cv)[0]

    # Check if rotation matrix and translation vector are correct
    assert np.allclose(
        rmatrix, rmatrix_cv, atol=1e-3
    ), f"Rotation matrix {rmatrix - rmatrix_cv} is incorrect"
    assert np.allclose(
        tvec, tvec_cv, atol=1e-3
    ), f"Translation vector {tvec - tvec_cv} is incorrect"


if __name__ == "__main__":
    pytest.main(["-v", "-s"])
