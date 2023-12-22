from typing import Any
import numpy as np

from vo.primitives import Frame, Features
from vo.helpers import to_homogeneous_coordinates, to_cartesian_coordinates


class Matches:
    """A class to represent the keypoint matches of 2 images"""

    def __init__(self, frame1: Frame, frame2: Frame, matches: np.ndarray):
        """Create two frames with only matching keypoints, descriptors and landmarks of the two images

        Args:
            frame1 (Frame): first object of class Frame.
            frame2 (Frame): second object of class Frame.
            matches (np.ndarray): array with indices of matching keypoints of the two images, shape = (M, 2).
        """

        self.frame1 = frame1
        self.frame2 = frame2

        self.newly_matched_idx = None
        self._threshold = 0.1

        # Get triangulated & matched keypoints
        triangulated_mask = self.frame1.features.state[matches[:, 0]] == 2
        matched_mask = self.frame1.features.state[matches[:, 0]] == 1
        newly_matched = self.frame1.features.state[matches[:, 0]] == 0

        triangulated_idx1 = matches[:, 0][triangulated_mask]
        matched_idx1 = matches[:, 0][matched_mask]
        newly_idx1 = matches[:, 0][newly_matched]
        unmatched_idx1 = np.delete(
            np.arange(0, len(self.frame1.features.keypoints)),
            np.concatenate([triangulated_idx1, matched_idx1, newly_idx1]),
        )

        kp1 = np.concatenate(
            [
                frame1.features.keypoints[triangulated_idx1],
                frame1.features.keypoints[matched_idx1],
                frame1.features.keypoints[newly_idx1],
                frame1.features.keypoints[unmatched_idx1],
            ],
            axis=0,
        )

        desc1 = (
            None
            if frame1.features.descriptors is None
            else np.concatenate(
                [
                    frame1.features.descriptors[triangulated_idx1],
                    frame1.features.descriptors[matched_idx1],
                    frame1.features.descriptors[newly_idx1],
                    frame1.features.descriptors[unmatched_idx1],
                ],
                axis=0,
            )
        )

        land1 = np.concatenate(
            [
                frame1.features.landmarks[triangulated_idx1],
                frame1.features.landmarks[matched_idx1],
                frame1.features.landmarks[newly_idx1],
                frame1.features.landmarks[unmatched_idx1],
            ],
            axis=0,
        )

        state1 = np.concatenate(
            (
                2 * np.ones_like(triangulated_idx1),
                1 * np.ones_like(matched_idx1),
                1 * np.ones_like(newly_idx1),
                0 * np.ones_like(unmatched_idx1),
            ),
        )

        assert not np.any(
            np.isnan(frame1.features.tracks[matched_idx1])
        ), "NaN in matched tracks"

        tracks1 = np.concatenate(
            (
                np.nan * np.ones(shape=(len(triangulated_idx1), 2, 1)),
                frame1.features.tracks[matched_idx1],
                frame1.features.keypoints[newly_idx1],
                np.nan * np.ones(shape=(len(unmatched_idx1), 2, 1)),
            ),
        )

        assert not np.any(
            np.isnan(frame1.features.poses[matched_idx1])
        ), "NaN in matched poses"
        # print(np.any(np.isnan(frame1.features.poses[matched_idx1])))

        poses1 = np.concatenate(
            (
                np.nan * np.ones(shape=(len(triangulated_idx1), 4, 4)),
                frame1.features.poses[matched_idx1],
                np.nan * np.ones(shape=(len(newly_idx1), 4, 4)),
                np.nan * np.ones(shape=(len(unmatched_idx1), 4, 4)),
            ),
        )

        assert not np.any(np.isnan(land1[state1 == 2])), "NaN in triangulated landmarks"

        self.frame1.features.keypoints = kp1
        self.frame1.features.state = state1
        self.frame1.features.descriptors = desc1
        self.frame1.features.landmarks = land1
        self.frame1.features.tracks = tracks1
        self.frame1.features.poses = poses1

        # Update features of frame2
        triangulated_idx2 = matches[:, 1][triangulated_mask]
        matched_idx2 = matches[:, 1][matched_mask]
        newly_idx2 = matches[:, 1][newly_matched]
        unmatched_idx2 = np.delete(
            np.arange(0, len(self.frame2.features.keypoints)),
            np.concatenate([triangulated_idx2, matched_idx2, newly_idx2]),
        )

        kp2 = np.concatenate(
            [
                frame2.features.keypoints[triangulated_idx2],
                frame2.features.keypoints[matched_idx2],
                frame2.features.keypoints[newly_idx2],
                frame2.features.keypoints[unmatched_idx2],
            ],
            axis=0,
        )

        desc2 = (
            None
            if frame1.features.descriptors is None
            else np.concatenate(
                [
                    frame2.features.descriptors[triangulated_idx2],
                    frame2.features.descriptors[matched_idx2],
                    frame2.features.descriptors[newly_idx2],
                    frame2.features.descriptors[unmatched_idx2],
                ],
                axis=0,
            )
        )

        land2 = np.concatenate(
            [
                self.frame1.features.landmarks[
                    : len(triangulated_idx2)
                ],  # copy triangulated landmarks from frame 1
                frame2.features.landmarks[
                    matched_idx2
                ],  # the other landmarks will be np.nan anyways
                frame2.features.landmarks[newly_idx2],
                frame2.features.landmarks[unmatched_idx2],
            ],
            axis=0,
        )

        state2 = np.concatenate(
            (
                2 * np.ones_like(triangulated_idx2),
                1 * np.ones_like(matched_idx2),
                1 * np.ones_like(newly_idx2),
                np.zeros_like(unmatched_idx2),
            )
        )

        tracks2 = np.concatenate(
            (
                np.nan * np.ones(shape=(len(triangulated_idx2), 2, 1)),
                self.frame1.features.tracks[
                    len(triangulated_idx2) : len(triangulated_idx2) + len(matched_idx2)
                ],
                self.frame1.features.tracks[
                    len(triangulated_idx2)
                    + len(matched_idx2) : len(triangulated_idx2)
                    + len(matched_idx2)
                    + len(newly_idx2)
                ],
                np.nan * np.ones(shape=(len(unmatched_idx2), 2, 1)),
            ),
        )

        poses2 = np.concatenate(
            (
                np.nan * np.ones(shape=(len(triangulated_idx2), 4, 4)),
                self.frame1.features.poses[
                    len(triangulated_idx2) : len(triangulated_idx2) + len(matched_idx2)
                ],
                np.nan * np.ones(shape=(len(newly_idx2), 4, 4)),
                np.nan * np.ones(shape=(len(unmatched_idx2), 4, 4)),
            ),
        )

        assert len(tracks2) == len(kp2), "Length of tracks and keypoints do not match"

        assert not np.any(np.isnan(land2[state2 == 2])), "NaN in triangulated landmarks"

        self.frame2.features.keypoints = kp2
        self.frame2.features.state = state2
        self.frame2.features.descriptors = desc2
        self.frame2.features.landmarks = land2
        self.frame2.features.tracks = tracks2
        self.frame2.features.poses = poses2

        self.newly_matched_idx = (
            [
                len(triangulated_idx1) + len(matched_idx1),
                len(triangulated_idx1) + len(matched_idx1) + len(newly_idx1),
            ],
            [
                len(triangulated_idx2) + len(matched_idx2),
                len(triangulated_idx2) + len(matched_idx2) + len(newly_idx2),
            ],
        )

    def set_pose(self, pose: np.ndarray) -> None:
        """Set the pose for the newly matched keypoints in both frames.

        Args:
            pose (np.ndarray): 4x4 transformation matrix representing the pose.
        """
        assert pose.shape == (4, 4), "Pose must be 4x4"
        self.frame1.features.poses[
            self.newly_matched_idx[0][0] : self.newly_matched_idx[0][1]
        ] = pose
        self.frame2.features.poses[
            self.newly_matched_idx[1][0] : self.newly_matched_idx[1][1]
        ] = pose

    def _to_camera_coordinates(self, K: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Converts pixel coordinates to camera coordinates.

        Args:
            K (np.ndarray): 3x3 camera intrinsic matrix.
            points (np.ndarray): Array of pixel coordinates with shape (N, 3, 1).

        Returns:
            np.ndarray: Array of camera coordinates with shape (N, 3, 1).
        """
        assert points.ndim == 3, "Points must have three dimensions"
        assert K.shape == (3, 3), "K must be 3x3 matrix"
        return np.linalg.inv(K) @ points

    def _to_world_coordinates_array(self, T, points: np.ndarray) -> np.ndarray:
        """Converts camera coordinates to world coordinates.

        Args:
            T (np.ndarray): Nx4x4 transformation matrix representing the camera pose.
            points (np.ndarray): Array of camera coordinates with shape (N, 4, 1).

        Returns:
            np.ndarray: Array of world coordinates with shape (N, 3, 1).
        """
        assert points.ndim == 3, "Points must have three dimensions"
        assert T.shape == (len(points), 4, 4), "T must be Nx4x4 matrix"
        transformed_points = np.einsum("ijk,ikl->ijl", T, points)
        return transformed_points[
            :,
            :3,
        ]

    def _to_world_coordinates(self, T, points: np.ndarray) -> np.ndarray:
        """Converts camera coordinates to world coordinates.

        Args:
            T (np.ndarray): 4x4 transformation matrix representing the camera pose.
            points (np.ndarray): Array of camera coordinates with shape (N, 4, 1).

        Returns:
            np.ndarray: Array of world coordinates with shape (N, 3, 1).
        """
        assert points.ndim == 3, "Points must have three dimensions"
        assert T.shape == (4, 4), "T must be Nx4x4 matrix"
        transformed_points = T @ points
        return transformed_points[
            :,
            :3,
        ]

    def _calculate_bearing_angle(self, K, T1, T2, points1, points2) -> np.ndarray:
        """Calculate the bearing angle between two points in two different frames.

        This method calculates the bearing angle between two points in two different frames using the camera intrinsic matrix,
        camera pose transformation matrices, and pixel coordinates of the points.

        Args:
            K (np.ndarray): 3x3 camera intrinsic matrix of the first frame.
            T1 (np.ndarray): Nx4x4 transformation matrix representing the camera pose of the first frame.
            T2 (np.ndarray): Nx4x4 transformation matrix representing the camera pose of the second frame.
            points1 (np.ndarray): Array of pixel coordinates with shape (N, 2, 1) in the first frame.
            points2 (np.ndarray): Array of pixel coordinates with shape (N, 2, 1) in the second frame.

        Returns:
            np.ndarray: Array of bearing angles with shape (N, 1).

        """
        assert len(points1) == len(points2), "Points must have same length"
        assert points1.ndim == 3, "Points must have three dimensions"
        assert points2.ndim == 3, "Points must have three dimensions"

        points1 = self._to_camera_coordinates(K, to_homogeneous_coordinates(points1))
        points2 = self._to_camera_coordinates(K, to_homogeneous_coordinates(points2))

        points1 = self._to_world_coordinates_array(
            T1, to_homogeneous_coordinates(points1)
        )
        points2 = self._to_world_coordinates(T2, to_homogeneous_coordinates(points2))

        return np.arccos(
            np.sum(points1 * points2, axis=-2)
            / (np.linalg.norm(points1, axis=-2) * np.linalg.norm(points2, axis=-2))
        )

    def set_candidate_mask(self, current_pose: np.ndarray, K: np.ndarray) -> None:
        """Set the candidate mask for the current frame.

        Args:
            current_pose (np.ndarray): pose of the current frame.
        """
        if current_pose.shape == (3, 4):
            current_pose = np.concatenate(
                (current_pose, np.array([[0, 0, 0, 1]])), axis=0
            )

        candidate_mask1 = np.zeros_like(self.frame1.features.candidate_mask).astype(
            bool
        )
        candidate_mask2 = np.zeros_like(self.frame1.features.candidate_mask).astype(
            bool
        )
        matched_keypoints1 = self.frame1.features.matched_candidate_inliers_tracks
        matched_keypoints2 = self.frame2.features.matched_candidate_inliers_keypoints
        matched_T1 = self.frame1.features.matched_candidate_inliers_poses

        # Calculate bearing angle
        bearing_angle = self._calculate_bearing_angle(
            K,
            matched_T1,
            current_pose,
            matched_keypoints1,
            matched_keypoints2,
        )

        matched_keypoints_idx1 = np.where(self.frame1.features.state == 1)[0]
        matched_keypoints_idx2 = np.where(self.frame1.features.state == 1)[0]
        candidate_keypoints_idx1 = np.where(bearing_angle > self._threshold)[0]
        candidate_keypoints_idx2 = np.where(bearing_angle > self._threshold)[0]
        candidate_mask1[matched_keypoints_idx1[candidate_keypoints_idx1]] = True
        candidate_mask2[matched_keypoints_idx2[candidate_keypoints_idx2]] = True

        self.frame1.features.candidate_mask = candidate_mask1
        self.frame2.features.candidate_mask = candidate_mask2
