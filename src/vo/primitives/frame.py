import numpy as np
import cv2

from vo.sensors import Camera
from vo.primitives import Features


class Frame:
    """A class to represent a single frame in a video sequence."""

    def __init__(
        self, image: np.ndarray, features: Features = None, sensor: Camera = None
    ):
        self.image = image
        self.frame_id = None
        self.features = features
        self.sensor = sensor

    def get_frame_id(self) -> int:
        """Return the frame id.

        Returns:
            int: The frame id.
        """
        return self.frame_id

    def show(self, other=None) -> None:
        """Show the image of the frame."""
        # show image with frame index
        if other is None:
            cv2.imshow("Frame {}".format(self.frame_id), self.image)
        else:
            # add some padding to the image
            img1 = cv2.copyMakeBorder(
                other.image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            img2 = cv2.copyMakeBorder(
                self.image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            # add description
            img = np.hstack((img1, img2))
            cv2.imshow("Frame {} and {}".format(other.frame_id, self.frame_id), img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_features(self) -> None:
        """Return a copy of the image with features drawn on it.

        Returns:
            np.array: A copy of the image with features drawn on it.
        """
        # TODO: Test this method. Could not test it because we don't have the features yet.
        img = cv2.drawKeypoints(
            self.image,
            self.features.keypoints,
            self.image,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        cv2.imshow("Frame {}".format(self.frame_id), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __next__(self):
        """Return the next frame in the dataset.

        Returns:
            Frame: The next frame in the dataset.
        """
        return self

    def __iter__(self):
        """Return an iterator over the dataset.

        Returns:
            DataLoader: An iterator over the dataset.
        """
        return self

    def __repr__(self) -> str:
        return "Frame id: {}".format(self.frame_id)
