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

        self.frame1 = frame1
        self.frame2 = frame2

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

        assert not np.any(np.isnan(land1[state1 == 2])), "NaN in triangulated landmarks"

        self.frame1.features.keypoints = kp1
        self.frame1.features.state = state1
        self.frame1.features.descriptors = desc1
        self.frame1.features.landmarks = land1

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

        assert not np.any(np.isnan(land2[state2 == 2])), "NaN in triangulated landmarks"

        self.frame2.features.keypoints = kp2
        self.frame2.features.state = state2
        self.frame2.features.descriptors = desc2
        self.frame2.features.landmarks = land2
