import numpy as np

from vo.helpers import to_cartesian_coordinates


class Camera:
    """
    A class to represent the camera with its intrinsic parameters and provide utility methods
    for image undistortion, projection, and pose estimation.
    """

    def __init__(
        self,
        intrinsic_matrix: np.ndarray,
        distortion_coeffs: np.ndarray = None,
        R: np.ndarray = None,
        t: np.ndarray = None,
    ):
        """
        Initialize the Camera class with intrinsic parameters.

        :param intrinsic_matrix: The camera's intrinsic matrix.
        :param distortion_coeffs: The camera's distortion coefficients.
        """
        self.intrinsic_matrix = intrinsic_matrix
        self.distortion_coeffs = distortion_coeffs
        self.R = R
        self.t = t

    @property
    def projection_matrix(self) -> np.ndarray:
        """
        :return: The focal length in the x direction.
        """
        assert self.R is not None and self.t is not None, "Camera pose not set"
        return self.intrinsic_matrix[0, 0] @ np.hstack((self.R, self.t))

    def distort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Distort points using the camera's distortion coefficients.

        :param points: An array of points.
        :return: The distorted points.
        """
        pass

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """
        Remove distortion from the image using the camera's distortion coefficients.

        :param image: A distorted image.
        :return: The undistorted image.
        """
        pass

    def project_points_world_frame(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points expressed in the world's frame into the camera's 2D image plane.

        :param points_3d: An array of 3D points in world frame coordinates.
        :return: An array of 2D points.
        """
        assert self.R is not None and self.t is not None, "Camera pose not set"
        points_camera_frame = self.R[np.newaxis] @ points_3d + self.t
        return self.project_points_camera_frame(points_camera_frame)

    def project_points_camera_frame(
        self,
        points_3d: np.ndarray,
    ) -> np.ndarray:
        """
        Project 3D points expressed in the camera's frame into the camera's 2D image plane.

        :param points_3d: An array of 3D points in camera frame coordinates.
        :return: An array of 2D points.
        """
        projections_hom = self.intrinsic_matrix[np.newaxis] @ points_3d
        return to_cartesian_coordinates(projections_hom)

    @property
    def c_T_w(self) -> np.ndarray:
        """
        :return: The transformation matrix from world to camera coordinates.
        """
        assert self.R is not None and self.t is not None, "Camera pose not set"
        return np.vstack((np.hstack((self.R, self.t)), [0, 0, 0, 1]))
