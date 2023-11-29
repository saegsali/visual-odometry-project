from typing import Tuple
import numpy as np
import cv2


class Camera:
    """
    A class to represent the camera with its intrinsic parameters and provide utility methods
    for image undistortion, projection, and pose estimation.
    """

    def __init__(
        self, intrinsic_matrix: np.ndarray, distortion_coeffs: np.ndarray = None
    ):
        """
        Initialize the Camera class with intrinsic parameters.

        :param intrinsic_matrix: The camera's intrinsic matrix.
        :param distortion_coeffs: The camera's distortion coefficients.
        """
        self.intrinsic_matrix = intrinsic_matrix
        self.distortion_coeffs = distortion_coeffs

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """
        Remove distortion from the image using the camera's distortion coefficients.

        :param image: A distorted image.
        :return: The undistorted image.
        """
        pass

    def project_points(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points into the camera's 2D image plane.

        :param points_3d: An array of 3D points.
        :return: An array of 2D points.
        """
        pass

    def estimate_pose(
        self, image_points: np.ndarray, world_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the camera pose given corresponding 2D image points and 3D world points.

        :param image_points: 2D points in the image plane.
        :param world_points: Corresponding 3D points in the world.
        :return: A tuple containing the rotation vector and translation vector.
        """
        pass
