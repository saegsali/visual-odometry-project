import numpy as np

from vo.primitives import Frame, Features


class Matches:
    """A class to represent the keypoint matches of 2 images"""

    def __init__(self, frame1: Frame, frame2: Frame, matches: np.ndarray):
        """Create two frames with only matching keypoints, descriptors and landmarks of the two images

        Args:
            frame1 (Frame): first object of class Frame.
            frame2 (Frame): second object of class Frame.
            matches (np.ndarray): array with indices of matching keypoints of the two images, shape = (M, 2).
        """

        self.frame1 = self.get_matching_keypoints(frame1, matches[:, 0], True)
        self.frame2 = self.get_matching_keypoints(frame2, matches[:, 1])

        if self.frame1.features.landmarks is not None:
            n_landmarks = (
                0
                if frame1.features.landmarks is None
                else frame1.features.landmarks.shape[0]
            )
            self.frame2.features.landmarks[
                :n_landmarks
            ] = self.frame1.features.landmarks[:n_landmarks]

    def get_matching_keypoints(self, frame: Frame, indices, check_landmarks=False):
        """Create frame with only matching keypoints, descriptors and landmarks

        Args:
            frame (Frame): object of class Frame.
            indices (np.ndarray): array with indices of matching keypoints of the image (with other image), shape = (M, 1).
        """
        assert (
            frame.features is not None and frame.features.keypoints is not None
        ), "Frame must have features with keypoints"

        if check_landmarks:
            n_landmarks = (
                0
                if frame.features.landmarks is None
                else frame.features.landmarks.shape[0]
            )
            mask = indices < n_landmarks
            triangulated_indices = indices[mask]
            candidate_indices = indices[~mask]
        else:
            triangulated_indices = []
            candidate_indices = indices

        kp = np.concatenate(
            [
                frame.features.keypoints[triangulated_indices],
                frame.features.keypoints[candidate_indices],
            ],
            axis=0,
        )

        desc = (
            None
            if frame.features.descriptors is None
            else np.concatenate(
                [
                    frame.features.descriptors[triangulated_indices],
                    frame.features.descriptors[candidate_indices],
                ],
                axis=0,
            )
        )

        lnd = (
            frame.features.landmarks
            if frame.features.landmarks is None
            else np.concatenate(
                [
                    frame.features.landmarks[triangulated_indices],
                    -np.ones(shape=(candidate_indices.shape[0], 3, 1)),
                ],
                axis=0,
            )
        )

        frame.features.update_features(keypoints=kp, descriptors=desc, landmarks=lnd)

        return frame

    def apply_inliers(self, inliers: np.ndarray) -> None:
        """Apply inliers to the matches.

        Args:
            inliers (np.ndarray): Boolean array of inliers.
        """
        self.frame1.features.apply_inliers(inliers)
        self.frame2.features.apply_inliers(inliers)

    def plot_matches(self):
        """Plot the the images with the matching keypoints"""
        # Todo
