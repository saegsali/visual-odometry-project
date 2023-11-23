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
        self.frame1 = self.get_matching_keypoints(frame1, matches[:, 0])
        self.frame2 = self.get_matching_keypoints(frame2, matches[:, 1])

    def get_matching_keypoints(self, frame, indices):
        """Create frame with only matching keypoints, descriptors and landmarks

        Args:
            frame (Frame): object of class Frame.
            indices (np.ndarray): array with indices of matching keypoints of the image (with other image), shape = (M, 1).
        """
        assert (
            frame.features is not None and frame.features.keypoints is not None
        ), "Frame must have features with keypoints"

        return Frame(
            image=frame.image,
            features=Features(
                keypoints=frame.features.keypoints[indices],
                descriptors=None
                if frame.features.descriptors is None
                else frame.features.descriptors[indices],
                landmarks=None
                if frame.features.landmarks is None
                else frame.features.landmarks[indices],
            ),
            sensor=frame.sensor,
        )

    def plot_matches(self):
        """Plot the the images with the matching keypoints"""
        # Todo
