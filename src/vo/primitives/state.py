import numpy as np
from vo.primitives import Frame, Matches
from vo.helpers import to_homogeneous_coordinates, to_cartesian_coordinates


class State:
    def __init__(self, initial_frame: Frame) -> None:
        self.curr_pose = np.eye(4)
        self.curr_frame = initial_frame

        self.prev_pose = None
        self.prev_frame = None

        # self.candidates = np.ones(shape=(self.curr_frame.features.length)).astype(
        #     bool
        # )  # use all as candidates for now

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

    def update_with_world_pose(self, pose: np.ndarray) -> None:
        """Update the state with new pose, which is relative to world frame (transforms world into frame).

        Args:
            pose (np.ndarray): relative transform from world to camera, given as 4x4 or 3x4 matrix
        """
        if pose.shape == (3, 4):
            pose = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0)
        self.curr_pose = np.linalg.inv(pose)

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

        self.curr_frame.features.landmarks[keypoints_mask] = landmarks
        self.curr_frame.features.state[keypoints_mask] = 2  # set to triangulated

    def get_frame(self) -> Frame:
        return self.curr_frame

    def get_pose(self) -> np.ndarray:
        return self.curr_pose

    def get_landmarks(self) -> np.ndarray:
        return self.curr_frame.features.landmarks

    def get_keypoints(self) -> np.ndarray:
        return self.curr_frame.features.keypoints

    def get_candidates(self) -> np.ndarray:
        raise NotImplementedError

    def set_match_inliers(self, inliers: np.ndarray) -> None:
        self.curr_frame.features.set_match_inliers(inliers)
        self.prev_frame.features.set_match_inliers(inliers)

    def set_p3p_inliers(self, inliers: np.ndarray) -> None:
        self.curr_frame.features.set_p3p_inliers(inliers)
        self.prev_frame.features.set_p3p_inliers(inliers)

    def set_triangulate_inliers(self, inliers: np.ndarray) -> None:
        self.curr_frame.features.set_triangulate_inliers(inliers)
        self.prev_frame.features.set_triangulate_inliers(inliers)
