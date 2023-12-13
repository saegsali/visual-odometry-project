import numpy as np
import cv2

from vo.helpers import (
    to_homogeneous_coordinates,
    to_cartesian_coordinates,
    normalize_points,
    to_skew_symmetric_matrix,
)
from vo.primitives import Features, Matches
from vo.algorithms import RANSAC
from vo.sensors import Camera
from vo.visualization import PointCloudVisualizer


class LandmarksTriangulator:
    def __init__(
        self,
        camera1: Camera,
        camera2: Camera,
        use_ransac: bool = True,
        outlier_ratio: float = 0.9,
        ransac_threshold: float = 3.0,
        ransac_confidence=0.99,
        use_opencv: bool = True,
    ) -> None:
        """Initializs the landmark triangulator and sets its parameters."""
        self.camera1 = camera1
        self.camera2 = camera2

        self._use_ransac = use_ransac
        self._outlier_ratio = outlier_ratio
        self._ransac_reproj_threshold = ransac_threshold
        self._ransac_confidence = ransac_confidence

        self._use_opencv = use_opencv

    def triangulate_matches(self, matches: Matches) -> np.ndarray:
        """Triangulate the 3D position of landmarks using matches from two frames,
        and as a byproduct, estimate the relative pose between frame 1 and frame 2.

        Args:
            matches (Matches): object containing the matches of each frame.

        Returns:
            np.ndarray: array of 3D landmark positions in world coordinates, shape = (N, 3, 1).
            np.ndarray: M = [R t] which transforms coordinates from camera frame 1 to camera frame 2, shape = (3, 4).
        """
        points1 = matches.frame1.features.keypoints
        points2 = matches.frame2.features.keypoints

        # Find relative pose
        if self._use_ransac:
            M, landmarks, inliers = self._find_relative_pose(points1, points2)
            return M, landmarks, inliers
        else:
            M, landmarks = self._find_relative_pose(points1, points2)
            return M, landmarks

    def _find_fundamental_matrix_ransac(
        self, points1: np.ndarray, points2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the fundamental matrix from the given points using RANSAC.

        Args:
            points1 (np.ndarray): array of 2D points/pixels in the first image, shape = (N, 2, 1).
            points2 (np.ndarray): array of 2D points/pixels in the second image, shape = (N, 2, 1).
            is_normalized (bool): if False, internally normalize points for better condition number. Defaults to False.

        Returns:
            np.ndarray: fundamental matrix, shape = (3, 3).
            np.ndarray: inlier mask
        """
        # Use OpenCV method
        if self._use_opencv:
            F, inliers = cv2.findFundamentalMat(
                points1=points1,
                points2=points2,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=self._ransac_reproj_threshold,
                confidence=self._ransac_confidence,
            )
            return F, inliers.astype(bool).flatten()

        def model_fn(population):
            return self._find_fundamental_matrix(
                population[:, 0], population[:, 1], is_normalized=True
            )

        def error_fn(F, points):
            # Compte algebraic error r =  p2^T @ F @ p1
            p1 = to_homogeneous_coordinates(points[:, 0])
            p2 = to_homogeneous_coordinates(points[:, 1])
            r = np.sum((p2.transpose((0, 2, 1)) @ F @ p1) ** 2, axis=(1, 2))
            return r

        # Normalize points for better condition number
        points1, T1 = normalize_points(points1)
        points2, T2 = normalize_points(points2)

        ransac_F = RANSAC(
            s_points=8,
            population=np.stack([points1, points2], axis=1),
            model_fn=model_fn,
            error_fn=error_fn,
            inlier_threshold=self._ransac_reproj_threshold,
            outlier_ratio=self._outlier_ratio,
            confidence=self._ransac_confidence,
        )

        F, inliers = ransac_F.find_best_model()

        return T2.T @ F @ T1, inliers

    def _find_fundamental_matrix(
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

        # OpenCV option
        if self._use_opencv:
            F, _ = cv2.findFundamentalMat(
                points1=points1, points2=points2, method=cv2.FM_8POINT
            )
            return F

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

    def _find_essential_matrix(
        self, points1: np.ndarray, points2: np.ndarray
    ) -> np.ndarray:
        """Find the essential matrix from the given points.

        Args:
            points1 (np.ndarray): array of 2D points/pixels in the first image, shape = (N, 2, 1).
            points2 (np.ndarray): array of 2D points/pixels in the second image, shape = (N, 2, 1).

        Returns:
            np.ndarray: essential matrix, shape = (3, 3).
        """
        if self._use_ransac:
            F, inliers = self._find_fundamental_matrix_ransac(points1, points2)
            E = self.camera2.intrinsic_matrix.T @ F @ self.camera1.intrinsic_matrix
            return E, inliers
        else:
            F = self._find_fundamental_matrix(points1, points2)
            E = self.camera2.intrinsic_matrix.T @ F @ self.camera1.intrinsic_matrix
            return E

    def _decompose_essential_matrix(self, E: np.ndarray) -> np.ndarray:
        """Decomposes the matrix E = [T]x @ R into translation (up to scale) and rotation.

        Args:
            E (np.ndarray): Essential matrix.

        Returns:
            np.ndarray: M = 4 * [R T] which transforms points from camera1 frame to camera2 frame.
                        Note that the four solutions must be disambiguated by triangulating points.
        """
        U, S, Vh = np.linalg.svd(E)

        # Translation (unit/direction vector)
        T = U[:, 2:]

        # Compute possible rotations
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        R = np.zeros((2, 3, 3))
        R[0] = U @ W @ Vh
        R[1] = U @ W.T @ Vh
        for i in range(2):
            if np.linalg.det(R[i]) < 0:
                R[i] *= -1

        # Compute possible transforms
        M = np.zeros((4, 3, 4))

        for i in range(2):
            for j in range(2):
                idx = 2 * i + j
                M[idx] = np.concatenate([R[j], (-1) ** i * T], axis=-1)

        return M

    def _find_relative_pose(
        self, points1: np.ndarray, points2: np.ndarray
    ) -> np.ndarray:
        """Computes the transformation matrix M which transforms points from frame 1 into frame 2.

        Args:
            points1 (np.ndarray): array of 2D points/pixels in the first image, shape = (N, 2, 1).
            points2 (np.ndarray): array of 2D points/pixels in the second image, shape = (N, 2, 1).

        Returns:
            np.ndarray: Matrix M = [R T] with shape (3, 4), which transforms points from frame 1 into frame 2.
            np.ndarray: triangulated points expressed in coordinates of frame 1, shape = (N, 3, 1).
            np.ndarray: inliers mask if use_ransac is set to True.
        """
        if self._use_ransac:
            E, inliers = self._find_essential_matrix(points1, points2)
            points1_inliers = points1[inliers]
            points2_inliers = points2[inliers]
        else:
            E = self._find_essential_matrix(points1, points2)
            points1_inliers = points1
            points2_inliers = points2

        # Find four possible solutions which need to be checked
        M2 = self._decompose_essential_matrix(
            E
        )  # relative transform from camera1 to camera2
        M1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # identity transform

        # Disambiguate relative pose by triangulating points and counting inliers
        best_valid = -1
        best_inliers = None
        best_M = None

        for m in range(M2.shape[0]):
            points3D = self._linear_triangulation(
                points1_inliers,
                points2_inliers,
                self.camera1.intrinsic_matrix @ M1,
                self.camera2.intrinsic_matrix @ M2[m],
            )
            points3D_frame2 = M2[m][:, :3] @ points3D + M2[m][:, 3:]

            # Count how many points are valid (in front of camera1 and camera2)
            cheirality_inliers1 = points3D[:, -1] >= 0
            cheirality_inliers2 = points3D_frame2[:, -1] >= 0
            cheirality_inliers = (cheirality_inliers1 & cheirality_inliers2).flatten()

            n_valid = cheirality_inliers.sum()

            if n_valid > best_valid:
                best_valid = n_valid
                best_inliers = cheirality_inliers
                best_M = M2[m]

        # Return for all 2D points triangulated points (even if outlier in previous steps)
        triangulated_points = self._linear_triangulation(
            points1,
            points2,
            self.camera1.intrinsic_matrix @ M1,
            self.camera2.intrinsic_matrix @ best_M,
        )

        # If RANSAC was used, return also the inliers mask
        if self._use_ransac:
            inlier_mask = np.zeros(
                (points1.shape[0],), dtype=bool
            )  # size of all inputs
            inlier_mask[inliers] = best_inliers
            return best_M, triangulated_points, inlier_mask

        return best_M, triangulated_points

    def _linear_triangulation(self, points1, points2, C1, C2):
        """Linear Triangulation
        Input:
        - p1 np.ndarray(3, N): homogeneous coordinates of points in image 1
        - p2 np.ndarray(3, N): homogeneous coordinates of points in image 2
        - C1 np.ndarray(3, 4): camera (projection matrix) corresponding to first image, shape (3, 4)
        - C2 np.ndarray(3, 4): camera (projection matrix) corresponding to second image, shape (3,4)

        Output:
        - P np.ndarray(4, N): homogeneous coordinates of 3-D points
        """

        assert points1.shape == points2.shape, "Input points dimension mismatch"
        assert points1.shape[1] == 2, "Points must have two rows for (u,v)"
        assert points1.shape[2] == 1, "Points must be a column vector"
        assert C1.shape == (3, 4) and C2.shape == (
            3,
            4,
        ), "Matrix C1 and C2 must be 3 rows and 4 columns [R T]"

        N = points1.shape[0]
        P = np.zeros((N, 4, 1))  # triangulated points

        points1 = to_homogeneous_coordinates(points1)
        points2 = to_homogeneous_coordinates(points2)

        # Triangulate all points
        for i in range(N):
            A1 = to_skew_symmetric_matrix(points1[i]) @ C1
            A2 = to_skew_symmetric_matrix(points2[i]) @ C2
            A = np.r_[A1, A2]

            # Solve the homogeneous system of equations
            assert A.shape[0] >= A.shape[1], "Underdetermined system of equations"
            _, _, Vh = np.linalg.svd(A, full_matrices=False)
            P[i, :] = Vh.T[:, -1:]

        return to_cartesian_coordinates(P)

    def visualize_fundamental_matrix(self, matches: Matches):
        """Visualize the computed fundamental matrix in the two frames by drawing
        the epilines and features.

        Args:
            matches (Matches): A matches containing the two frames to visualized.
        """

        F, inliers = self._find_fundamental_matrix_ransac(
            matches.frame1.features.keypoints, matches.frame2.features.keypoints
        )

        img1 = matches.frame1.image
        img2 = matches.frame2.image

        # Compute line parameters in left and right image
        lines1 = F @ to_homogeneous_coordinates(
            matches.frame1.features.keypoints[inliers]
        )
        lines2 = F.T @ to_homogeneous_coordinates(
            matches.frame2.features.keypoints[inliers]
        )  # [a b c] @ [x y 1] = 0 -> y = -a/b * x - c/b

        # Draw images next to each other
        if img1.ndim < 3 or img1.shape[2] == 1:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        if img2.ndim < 3 or img2.shape[2] == 1:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

        # Draw epilines
        def draw_epilines(img, lines, colors=None):
            img = img.copy()
            H, W = img.shape[:2]

            if colors is None:
                colors = np.random.randint(0, 255, (lines.shape[0], 3))

            for i, line in enumerate(lines):
                # Compute two points on line
                line = line.flatten()
                p0 = np.array([0, -line[2] / line[1]]).astype(int)
                p1 = np.array([W, -line[0] / line[1] * W - line[2] / line[1]]).astype(
                    int
                )
                img = cv2.line(img, p0, p1, tuple(colors[i].tolist()))

            return img

        assert lines1.shape == lines2.shape
        colors = np.random.randint(0, 255, (lines1.shape[0], 3))
        img1 = draw_epilines(img1, lines2[:50], colors=colors)
        img2 = draw_epilines(img2, lines1[:50], colors=colors)

        img_both = cv2.hconcat([img1, img2])

        cv2.imshow("Epilines", img_both)

    def visualize_triangulation(
        self, matches: Matches, visualizer: PointCloudVisualizer = None
    ):
        """Visualize the triangulated points from matches.

        Args:
            matches (Matches): A matches containing the two frames to visualized.
        """
        M, points3D, inliers, _ = self.triangulate_matches(matches)

        camera1 = Camera(
            intrinsic_matrix=self.camera1.intrinsic_matrix,
            R=np.eye(3),
            t=np.zeros((3, 1)),
        )
        camera2 = Camera(
            intrinsic_matrix=self.camera2.intrinsic_matrix, R=M[:, :3], t=M[:, 3:]
        )

        if visualizer is None:
            visualizer = PointCloudVisualizer()
        visualizer.visualize_camera(camera1)
        visualizer.visualize_camera(camera2)
        visualizer.visualize_points(points3D[inliers], color="g")
        visualizer.visualize_points(points3D[~inliers], color="r")

        return visualizer


# Example to visualze estimated fundamental matrix using epilines
if __name__ == "__main__":
    from vo.primitives.loader import Sequence

    # Create visualizer
    visualizer = PointCloudVisualizer()

    # Load sequence
    sequence = Sequence("malaga", camera=1, use_lowres=True)
    frame1 = next(sequence)

    for frame2 in sequence:
        print(frame2)
        # Create camera
        camera = Camera(
            intrinsic_matrix=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
        )

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
        triangulator = LandmarksTriangulator(camera1=camera, camera2=camera)
        triangulator.visualize_fundamental_matrix(matches)

        # Store current frame as old frame
        frame1 = frame2

        # Visualize triangulation
        triangulator.visualize_triangulation(matches, visualizer=visualizer)
        key = cv2.waitKey(250)
