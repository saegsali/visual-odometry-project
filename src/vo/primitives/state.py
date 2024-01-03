import numpy as np
from vo.primitives import Frame, Matches
from vo.helpers import to_homogeneous_coordinates, to_cartesian_coordinates
from vo.sensors import Camera


class State:
    def __init__(self, initial_frame: Frame, bearing_threshold: float = 0.0075) -> None:
        self.curr_pose = np.eye(4)
        self.curr_frame = initial_frame

        self.prev_pose = None
        self.prev_frame = None

        self._bearing_threshold = bearing_threshold

    def update_from_matches(self, matches: Matches) -> None:
        self.prev_frame = self.curr_frame
        self.prev_pose = self.curr_pose

        self.curr_frame = matches.frame2
        self.curr_pose = None

    def update_with_local_pose(self, pose: np.ndarray) -> None:
        """Update the state with new pose, which is relative to frame of previous camera.

        Args:
            pose (np.ndarray): relative transform from camera 1 to camera 2, gives as 4x4 or 3x4 matrix
        """
        if pose.shape == (3, 4):
            pose = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0)
        self.curr_pose = self.prev_pose @ np.linalg.inv(pose)

        # Update pose of tracks in current frame
        self.curr_frame.features.set_pose_for_new_tracks(self.curr_pose)

    def update_with_world_pose(self, pose: np.ndarray) -> None:
        """Update the state with new pose, which is relative to world frame (transforms world into frame).

        Args:
            pose (np.ndarray): relative transform from world to camera, given as 4x4 or 3x4 matrix
        """
        if pose.shape == (3, 4):
            pose = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0)
        self.curr_pose = np.linalg.inv(pose)

        # Update pose of tracks in current frame
        self.curr_frame.features.set_pose_for_new_tracks(self.curr_pose)

    def update_with_local_landmarks(
        self,
        landmarks: np.ndarray,
        keypoints_mask: np.ndarray,
    ) -> None:
        """Update the state with new landmarks, which are local frame of previous camera.

        Args:
            landmarks (np.ndarray): List of landmarks.
        """

        landmarks = to_cartesian_coordinates(
            self.prev_pose @ to_homogeneous_coordinates(landmarks)
        )

        self.update_with_world_landmarks(landmarks, keypoints_mask)

    def update_with_world_landmarks(
        self, landmarks: np.ndarray, keypoints_mask: np.ndarray
    ) -> None:
        """Update the state with new landmarks, which are in world frame.

        Args:
            landmarks (np.ndarray): List of landmarks.
        """
        assert np.sum(keypoints_mask) == len(landmarks), "Mismatch in length"
        assert np.all(
            self.curr_frame.features.state[keypoints_mask] == 1
        ), "Already triangulated point"

        self.curr_frame.features.landmarks[keypoints_mask] = landmarks
        self.curr_frame.features.state[keypoints_mask] = 2  # set to triangulated
        self._check_landmarks()

        assert not np.any(
            np.isnan(
                self.curr_frame.features.landmarks[self.curr_frame.features.state == 2]
            )
        ), "NaN in triangulated landmarks"

    def _check_landmarks(self) -> None:
        # check if a landmark is behind the camera or not
        # convert landmarks to camera coordinates
        camera_landmarks_curr = to_cartesian_coordinates(
            np.linalg.inv(self.curr_pose)
            @ to_homogeneous_coordinates(self.curr_frame.features.landmarks)
        )
        camera_landmarks_prev = to_cartesian_coordinates(
            np.linalg.inv(self.prev_pose)
            @ to_homogeneous_coordinates(self.curr_frame.features.landmarks)
        )
        outliers = (camera_landmarks_curr[:, 2].flatten() < 0) | (
            camera_landmarks_prev[:, 2].flatten() < 0
        )
        self.curr_frame.features.landmarks[outliers] = np.nan
        self.reset_outliers(outliers)

        # # remove landmarks which are too far away
        # outliers = (np.linalg.norm(camera_landmarks_curr, axis=1) > 1000).flatten()
        # self.curr_frame.features.landmarks[outliers] = np.nan
        # self.reset_outliers(outliers)

    def get_frame(self) -> Frame:
        return self.curr_frame

    def get_pose(self) -> np.ndarray:
        return self.curr_pose

    def get_landmarks(self) -> np.ndarray:
        return self.curr_frame.features.landmarks

    def get_keypoints(self) -> np.ndarray:
        return self.curr_frame.features.keypoints

    def compute_candidates(self) -> None:
        # Get matched keypoints which are not yet triangulated
        keypoints_end = (
            self.curr_frame.features.matched_candidate_inliers_keypoints
        )  # end of track
        keypoints_start = (
            self.curr_frame.features.matched_candidate_inliers_tracks
        )  # start of track

        pose_start = self.curr_frame.features.matched_candidate_inliers_poses
        pose_end = np.stack([self.curr_pose] * pose_start.shape[0], axis=0)

        # Compute the bearing angle for all tracks
        bearing_angles = self._calculate_bearing_angle(
            self.curr_frame.sensor,
            pose_start,
            pose_end,
            keypoints_start,
            keypoints_end,
        )

        # Set candidates
        candidates = bearing_angles >= self._bearing_threshold
        self.curr_frame.features.candidate_mask[
            self.curr_frame.features.matched_candidate_inliers
        ] = candidates

    def reset_outliers(self, outliers: np.ndarray) -> None:
        """Resets all outliers to unmatched state and resets track and pose of track.

        Args:
            outliers (np.ndarray): outlier mask for all keypoints
        """
        self.curr_frame.features.state[outliers] = 0  # unmatched state
        self.curr_frame.features.tracks[outliers] = self.curr_frame.features.keypoints[
            outliers
        ]
        self.curr_frame.features.poses[outliers] = self.curr_pose

    def _calculate_bearing_angle(
        self,
        camera: Camera,
        T1: np.ndarray,
        T2: np.ndarray,
        points1: np.ndarray,
        points2: np.ndarray,
    ) -> np.ndarray:
        """Calculate the bearing angle between two points in two different frames.

        This method calculates the bearing angle between two points in two different frames using the camera intrinsic matrix,
        camera pose transformation matrices, and pixel coordinates of the points.

        Args:
            camera (np.ndarray):  Sensor object of camera (note: assumes that at all track positions same camera used).
            T1 (np.ndarray): Nx4x4 transformation matrix representing the camera pose of the start of the track.
            T2 (np.ndarray): Nx4x4 transformation matrix representing the camera pose of the end of the track.
            points1 (np.ndarray): Array of pixel coordinates with shape (N, 2, 1) at the start of the track.
            points2 (np.ndarray): Array of pixel coordinates with shape (N, 2, 1) at the end of the track.

        Returns:
            np.ndarray: Array of bearing angles with shape (N, ).

        """
        assert len(points1) == len(points2), "Points must have same length"
        assert points1.ndim == 3, "Points must have three dimensions"
        assert points2.ndim == 3, "Points must have three dimensions"

        assert not np.any(np.isnan(points1)), "Points1 contains invalid points"
        assert not np.any(np.isnan(points2)), "Points2 contains invalid points"
        assert not np.any(np.isnan(T1)), "Invalid start poses"
        assert not np.any(np.isnan(T2)), "Invalid end poses"

        dir1 = camera.to_normalized_image_coordinates(points1)
        dir2 = camera.to_normalized_image_coordinates(points2)

        # Rotate both direction / bearing bectors into world frame
        dir1 = np.matmul(T1[:, :3, :3], dir1).reshape(-1, 3)
        dir2 = np.matmul(T2[:, :3, :3], dir2).reshape(-1, 3)

        # Compute angle between vectors
        angles = np.arccos(
            np.sum(dir1 * dir2, axis=-1)
            / (np.linalg.norm(dir1, axis=-1) * np.linalg.norm(dir2, axis=-1))
        )
        return angles
