import numpy as np
import cv2

from vo.helpers import (
    to_cartesian_coordinates,
    to_homogeneous_coordinates,
    normalize_points,
)
from vo.primitives import Features, Frame, Matches


class LandmarksTriangulator:
    def __init__(
        self, ransac_reproj_threshold: float = 3.0, ransac_confidence=0.99
    ) -> None:
        """Initializs the landmark triangulator and sets its parameters."""
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.ransac_confidence = ransac_confidence

    def triangulate_matches(self, matches: Matches) -> np.ndarray:
        """Triangulate the 3D position of landmarks using matches from two frames.

        Args:
            matches (Matches): object containing the matches of each frame.

        Returns:
            np.ndarray: array of 3D landmark positions in world coordinates, shape = (N, 3).
        """
        points1 = matches.frame1.features.keypoints
        points2 = matches.frame2.features.keypoints

    def find_fundamental_matrix(
        self, points1: np.ndarray, points2: np.ndarray, is_normalized: bool = False
    ) -> np.ndarray:
        """Find the fundamental matrix from the given points.

        Args:
            points1 (np.ndarray): array of 2D points/pixels in the first image, shape = (N, 2).
            points2 (np.ndarray): array of 2D points/pixels in the second image, shape = (N, 2).
            is_normalized (bool): if False, internally normalize points for better condition number. Defaults to False.

        Returns:
            np.ndarray: fundamental matrix, shape = (3, 3).
        """
        assert points1.shape == points2.shape, "Input points dimension mismatch"
        assert points1.shape[1] == 2, "Points must have two columns for (u,v)"
        assert points1.shape[0] >= 8, "Not enough points for 8-point algorithm"

        # Useful variables
        N = points1.shape[0]

        # Normalize for better condition
        if not is_normalized:
            points1, T1 = normalize_points(points1)
            points2, T2 = normalize_points(points2)

        # Convert to homoegenous coordinates
        points1 = to_homogeneous_coordinates(points1)
        points2 = to_homogeneous_coordinates(points2)

        # Compute Q
        Q = np.empty(shape=(N, 3 * 3))
        for i in range(N):
            Q[i] = np.kron(points1[i], points2[i])

        # Perform SVD / least-squares estimation
        assert Q.shape[0] >= Q.shape[1], "Underdetermined system of equations"
        _, _, Vh = np.linalg.svd(
            Q, full_matrices=False
        )  # overdetermined case, V is fully computed anyways
        F = Vh[-1, :].reshape(3, 3).T  # stacked column-wise

        # Enforce det(F) = 0 by setting the smallest singular value to zero
        U, S, Vh = np.linalg.svd(F)
        S[-1] = 0
        F = U @ np.diag(S) @ Vh

        # Renormalize if required
        if not is_normalized:
            F = T2.T @ F @ T1
            print(T1, T2)

        return F
