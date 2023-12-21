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

        # Get triangulated & matched keypoints
        triangulated_mask = frame1.features.state[matches[:, 0]] == 2
        matched_mask = frame1.features.state[matches[:, 0]] == 1
        newly_matched = frame1.features.state[matches[:, 0]] == 0

        self.frame1 = self.rearrange_featuers_from_matches(
            frame1, matches[:, 0], triangulated_mask, matched_mask, newly_matched
        )
        self.frame2 = self.rearrange_featuers_from_matches(
            frame2, matches[:, 1], triangulated_mask, matched_mask, newly_matched
        )

    def rearrange_featuers_from_matches(
        self,
        frame: Frame,
        matches: np.ndarray,
        triangulated_mask: np.ndarray,
        matched_mask: np.ndarray,
        newly_matched: np.ndarray,
    ) -> Frame:
        """Rearrange the features of a frame according to the matches."""
        triangulated_idx = matches[triangulated_mask]
        matched_idx = matches[matched_mask]
        newly_idx = matches[newly_matched]
        unmatched_idx = np.delete(
            np.arange(0, len(frame.features.keypoints)),
            np.concatenate([triangulated_idx, matched_idx, newly_idx]),
        )

        kp = np.concatenate(
            [
                frame.features.keypoints[triangulated_idx],
                frame.features.keypoints[matched_idx],
                frame.features.keypoints[newly_idx],
                frame.features.keypoints[unmatched_idx],
            ],
            axis=0,
        )

        desc = (
            None
            if frame.features.descriptors is None
            else np.concatenate(
                [
                    frame.features.descriptors[triangulated_idx],
                    frame.features.descriptors[matched_idx],
                    frame.features.descriptors[newly_idx],
                    frame.features.descriptors[unmatched_idx],
                ],
                axis=0,
            )
        )

        land = np.concatenate(
            [
                frame.features.landmarks[triangulated_idx],
                frame.features.landmarks[matched_idx],
                frame.features.landmarks[newly_idx],
                frame.features.landmarks[unmatched_idx],
            ],
            axis=0,
        )

        state = np.concatenate(
            (
                2 * np.ones_like(triangulated_idx),
                1 * np.ones_like(matched_idx),
                1 * np.ones_like(newly_idx),
                0 * np.ones_like(unmatched_idx),
            ),
        )

        frame.features.keypoints = kp
        frame.features.state = state
        frame.features.descriptors = desc
        frame.features.landmarks = land

        return frame
