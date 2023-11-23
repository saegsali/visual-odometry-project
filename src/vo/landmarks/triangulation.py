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
            np.ndarray: array of 3D landmark positions in world coordinates, shape = (N, 3, 1).
        """
        points1 = matches.frame1.features.keypoints
        points2 = matches.frame2.features.keypoints

    def find_fundamental_matrix(
        self, points1: np.ndarray, points2: np.ndarray, is_normalized: bool = False
    ) -> np.ndarray:
        """Find the fundamental matrix from the given points.

        Args:
            points1 (np.ndarray): array of 2D points/pixels in the first image, shape = (N, 2, 1).
            points2 (np.ndarray): array of 2D points/pixels in the second image, shape = (N, 2, 1).
            is_normalized (bool): if False, internally normalize points for better condition number. Defaults to False.

        Returns:
            np.ndarray: fundamental matrix, shape = (3, 3).
        """
        assert points1.shape == points2.shape, "Input points dimension mismatch"
        assert points1.shape[0] >= 8, "Not enough points for 8-point algorithm"
        assert points1.shape[1] == 2, "Points must have two rows for (u,v)"
        assert points1.shape[2] == 1, "Points must be a column vector"

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
            Q[i] = np.kron(points1[i], points2[i]).T

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

    def visualize_fundamental_matrix(self, matches: Matches):
        """Visualize the computed fundamental matrix in the two frames by drawing
        the epilines and features.

        Args:
            matches (Matches): A matches containing the two frames to visualized.
        """

        F = self.find_fundamental_matrix(
            matches.frame1.features.keypoints, matches.frame2.features.keypoints
        )

        img1 = matches.frame1.image
        img2 = matches.frame2.image

        # Draw images next to each other
        if img1.ndim < 3 or img1.shape[2] == 1:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        if img2.ndim < 3 or img2.shape[2] == 1:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

        # Draw epilines
