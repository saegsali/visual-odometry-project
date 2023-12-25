import numpy as np
import cv2
import matplotlib.pyplot as plt

from vo.primitives import Features, Matches, Sequence
from vo.algorithms import RANSAC
from vo.sensors import Camera
from vo.landmarks import LandmarksTriangulator
from vo.helpers import to_homogeneous_coordinates


class P3PPoseEstimator:
    def __init__(
        self,
        intrinsic_matrix: np.ndarray,
        inlier_threshold: float,
        use_opencv: bool = True,
        outlier_ratio: float = 0.9,
        confidence: float = 0.99,
        max_iterations: int = 10000,
    ) -> None:
        """Initializes a P3P pose estimator.

        Args:
            intrinsic_matrix (np.ndarray): The intrinsic matrix of the camera.
            inlier_threshold (float): The threshold for the error term to be considered an inlier.
            outlier_ratio (float, optional): Percentage of estimated outlier points, serves as an upper bound for the adaptive case. Defaults to 0.9.
            confidence (float, optional): Confidence level that all inliers are selected. Defaults to 0.99.
            max_iterations (int, optional): Maximum number of iterations the algorithm should perform before stopping. Defaults to np.inf.
        """
        self._use_opencv = use_opencv
        self.intrinsic_matrix = intrinsic_matrix
        self.inlier_threshold = inlier_threshold
        self.outlier_ratio = outlier_ratio
        self.confidence = confidence
        self.max_iterations = max_iterations

        def plot_2Dpoints(points: np.ndarray) -> None:
            fig, ax = plt.subplots()
            for i, pointArray in enumerate(points):
                color = "r" if i == 0 else "g"
                label = "keypoints" if i == 0 else "projected points"
                ax.scatter(pointArray[:, 0], pointArray[:, 1], color=color, label=label)
            plt.legend()
            plt.show()

        def model_fn(
            points: np.ndarray, K: np.ndarray = self.intrinsic_matrix
        ) -> tuple | None:
            """
            Estimate the camera pose using the Perspective-n-Point (PnP) algorithm.

            Args:
                points (np.ndarray): Array of 3D-2D point correspondences.
                K (np.ndarray, optional): Intrinsic camera matrix. Defaults to self.intrinsic_matrix.

            Returns:
                tuple | None: Tuple containing the estimated rotation matrix and translation vector
                              which transfroms points in 3D coordinate to camera frame if successful,
                              otherwise None.

            """
            assert points.shape[0] == 4, "P3P requires 4 point correspondences"

            success, rvec, tvec = cv2.solvePnP(
                np.stack(points[:, 0]),  # landmarks
                np.stack(points[:, 1]),  # keypoints
                K,
                None,
                flags=cv2.SOLVEPNP_P3P,
            )
            if success:
                rotation_matrix = cv2.Rodrigues(rvec)[0]
                return rotation_matrix, tvec
            return None

        def error_fn(
            model: tuple,
            population: np.ndarray | list | tuple,
            K: np.ndarray = self.intrinsic_matrix,
        ) -> np.ndarray:
            """
            Calculates the error between the projected points and the keypoints.

            Args:
                model (tuple): Tuple containing the guess for the camera rotation matrix (R_C_W_guess) and translation (t_C_W_guess).
                population (np.ndarray | list | tuple): Array-like object containing the landmarks and keypoints.
                K (np.ndarray, optional): Intrinsic matrix of the camera. Defaults to self.intrinsic_matrix.

            Returns:
                np.ndarray: Array of errors between the keypoints and the projected points.
            """
            R_C_W_guess = model[0]
            t_C_W_guess = model[1]
            landmarks = np.stack(population[:, 0])
            keypoints = np.stack(population[:, 1])
            projected_points, _ = cv2.projectPoints(
                landmarks, R_C_W_guess, t_C_W_guess, K, None
            )
            projected_points = projected_points.reshape((-1, 2, 1))
            # plot_2Dpoints(np.array([keypoints, projected_points]))
            difference = keypoints - projected_points
            errors = np.linalg.norm(difference, axis=(1, 2)) ** 2
            return errors

        # initialize RANSAC
        self.ransac = RANSAC(
            s_points=4,
            population=None,
            model_fn=model_fn,
            error_fn=error_fn,
            inlier_threshold=self.inlier_threshold,
            outlier_ratio=self.outlier_ratio,
            confidence=self.confidence,
            max_iterations=self.max_iterations,
            p3p=True,
        )

    def estimate_pose(self, features: Features) -> tuple[object, np.ndarray]:
        """Estimates the pose of the camera using the P3P algorithm.

        Args:
            features (Features): Object containing the 3D landmarks and 2D keypoints.

        Returns:
            object: Rotation matrix and translation vector of the camera pose
                    which transfrom points in 3D coordinate to camera frame.
            np.ndarray: Boolean array of inliers.
        """
        points_3d = features.landmarks
        points_2d = features.keypoints

        assert points_3d.shape[1] == 3 and points_2d.shape[1] == 2, "Invalid shape."
        assert (
            points_3d is not None and points_2d is not None
        ), "3D landmarks and 2D keypoints must be provided."

        if self._use_opencv:
            success, rvec_cv, tvec_cv, inliers_cv = cv2.solvePnPRansac(
                points_3d,
                points_2d,
                self.intrinsic_matrix,
                None,
                flags=cv2.SOLVEPNP_P3P,
                iterationsCount=self.max_iterations,
                reprojectionError=self.inlier_threshold,
                confidence=self.confidence,
            )
            assert success, "OpenCV P3P failed"
            rmatrix_cv = cv2.Rodrigues(rvec_cv)[0]
            inliers_mask = np.zeros(shape=(features.length,), dtype=bool)
            inliers_mask[inliers_cv] = True
            return (rmatrix_cv, tvec_cv), inliers_mask

        N = features.length

        population = np.empty((N, 2), dtype=object)
        for i in range(N):
            population[i, 0] = points_3d[i]
            population[i, 1] = points_2d[i]

        # Find best model
        best_model, best_inlier = self.ransac.find_best_model(population=population)

        # Return the best p3p model and its inliers
        return best_model, best_inlier


if __name__ == "__main__":
    # Load sequence
    sequence = Sequence("malaga", camera=1, use_lowres=True)
    frame1 = next(sequence)
    frame2 = next(sequence)
    camera = sequence.get_camera()
    triangulator = LandmarksTriangulator(
        camera1=camera, camera2=camera, use_opencv=True
    )
    pose_estimator = P3PPoseEstimator(
        intrinsic_matrix=camera.intrinsic_matrix,
        inlier_threshold=1,
        outlier_ratio=0.9,
        confidence=0.99,
        max_iterations=1000,
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
    M, landmarks, inliers = triangulator.triangulate_matches(matches)
    matches.frame2.features.landmarks = landmarks
    matches.frame2.features.apply_inliers(inliers)
    model, inliers = pose_estimator.estimate_pose(matches.frame2.features)
    rmatrix, tvec = model
    # compare rmatrix and tvec with ground truth
    print("3D-2D Rotation matrix: ", rmatrix)
    print("2D-2D rotation matrix: ", M[:3, :3])
    print("3D-2D Translation vector: ", tvec[:, 0])
    print("2D-2D translation vector: ", M[:3, 3])
