import numpy as np
import cv2

from vo.helpers import (
    to_cartesian_coordinates,
    to_homogeneous_coordinates,
    normalize_points,
)
from vo.primitives import Features, Frame, Matches
from vo.algorithms import RANSAC


class LandmarksTriangulator:
    def __init__(
        self,
        outlier_ratio: float = 0.5,
        ransac_threshold: float = 3.0,
        ransac_confidence=0.99,
    ) -> None:
        """Initializs the landmark triangulator and sets its parameters."""
        self.outlier_ratio = outlier_ratio
        self.ransac_reproj_threshold = ransac_threshold
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

        # TODO: Use find_fundamental_matrix with RANSAC
        # Normalize points ones and use is_normalized = True to prevent from normalizing each time
        # Rescale estimated F to account for normalization
        pass

    def find_fundamental_matrix_ransac(
        self, points1: np.ndarray, points2: np.ndarray
    ) -> np.ndarray:
        """Find the fundamental matrix from the given points using RANSAC.

        Args:
            points1 (np.ndarray): array of 2D points/pixels in the first image, shape = (N, 2, 1).
            points2 (np.ndarray): array of 2D points/pixels in the second image, shape = (N, 2, 1).
            is_normalized (bool): if False, internally normalize points for better condition number. Defaults to False.

        Returns:
            np.ndarray: fundamental matrix, shape = (3, 3).
        """
        ransac_find_fundamental = lambda x: self.find_fundamental_matrix(
            x[:, 0], x[:, 1], is_normalized=True
        )

        # Normalize points for better condition number
        points1, T1 = normalize_points(points1)
        points2, T2 = normalize_points(points2)

        def error_fn(F, points):
            # Compte algebraic error r =  p2^T @ F @ p1
            p1 = to_homogeneous_coordinates(points[:, 0])
            p2 = to_homogeneous_coordinates(points[:, 1])
            r = np.sum((p2.transpose((0, 2, 1)) @ F @ p1) ** 2, axis=(1, 2))
            return r

        ransac_F = RANSAC(
            s_points=8,
            population=np.stack([points1, points2], axis=1),
            model_fn=ransac_find_fundamental,
            error_fn=error_fn,
            inlier_threshold=self.ransac_reproj_threshold,
            outlier_ratio=self.outlier_ratio,
            confidence=self.ransac_confidence,
        )

        F, inliers = ransac_F.find_best_model()

        return T2.T @ F @ T1, inliers

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
        # assert Q.shape[0] >= Q.shape[1], "Underdetermined system of equations"
        _, _, Vh = np.linalg.svd(
            Q, full_matrices=True
        )  # overdetermined case, V is fully computed anyways
        F = Vh[-1, :].reshape(3, 3).T  # stacked column-wise

        # Enforce det(F) = 0 by setting the smallest singular value to zero
        U, S, Vh = np.linalg.svd(F)
        S[-1] = 0
        F = U @ np.diag(S) @ Vh

        # Renormalize if required
        if not is_normalized:
            F = T2.T @ F @ T1
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

        # Compute line parameters in left and right image
        lines1 = F @ to_homogeneous_coordinates(matches.frame1.features.keypoints)
        lines2 = F.T @ to_homogeneous_coordinates(
            matches.frame2.features.keypoints
        )  # [a b c] @ [x y 1] = 0 -> y = -a/b * x - c/b

        # Draw images next to each other
        if img1.ndim < 3 or img1.shape[2] == 1:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        if img2.ndim < 3 or img2.shape[2] == 1:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

        # Draw epilines
        def draw_epilines(img, lines, colors=None):
            H, W = img.shape[:2]

            if colors is None:
                colors = np.random.randint(0, 255, (lines.shape[0], 3))

            for i, l in enumerate(lines):
                # Compute two points on line
                l = l.flatten()
                p0 = np.array([0, -l[2] / l[1]]).astype(int)
                p1 = np.array([W, -l[0] / l[1] * W - l[2] / l[1]]).astype(int)
                img = cv2.line(img, p0, p1, tuple(colors[i].tolist()))

            return img

        assert lines1.shape == lines2.shape
        colors = np.random.randint(0, 255, (lines1.shape[0], 3))
        img1 = draw_epilines(img1, lines2[:50], colors=colors)
        img2 = draw_epilines(img2, lines1[:50], colors=colors)

        img_both = cv2.hconcat([img1, img2])

        cv2.imshow("Epilines", img_both)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Example to visualze estimated fundamental matrix using epilines
if __name__ == "__main__":
    from vo.primitives.loader import Sequence

    # Load sequence
    sequence = Sequence("malaga", camera=1, use_lowres=True)
    frame1 = sequence.get_frame(0)
    frame2 = sequence.get_frame(1)

    # Compute matches using SIFT (TODO: replace later with our FeatureMatcher)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(frame1.image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(frame2.image, None)

    keypoints1 = np.array([kp.pt for kp in keypoints1]).reshape(-1, 2, 1)
    keypoints2 = np.array([kp.pt for kp in keypoints2]).reshape(-1, 2, 1)
    descriptors1 = np.array(descriptors1)
    descriptors2 = np.array(descriptors2)

    frame1.features = Features(keypoints1, descriptors1)
    frame2.features = Features(keypoints2, descriptors2)

    # Match keypoints
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m.queryIdx, m.trainIdx])
    good = np.array(good)

    # Visualize fundamental matrix
    matches = Matches(frame1, frame2, matches=good)
    triangulator = LandmarksTriangulator()
    triangulator.visualize_fundamental_matrix(matches)
