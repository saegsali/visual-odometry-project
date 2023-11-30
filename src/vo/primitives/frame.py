import numpy as np
import cv2 as cv

from vo.sensors import Camera
from vo.primitives import Features


class Frame:
    """A class to represent a single frame in a video sequence."""

    def __init__(
        self,
        image: np.ndarray,
        features: Features = None,
        sensor: Camera = None,
        intrinsics: np.ndarray = None,
    ):
        self.image = image
        self.frame_id = None
        self.features = features
        self.intrinsics = intrinsics
        self.sensor = sensor

    def get_frame_id(self) -> int:
        """Return the frame id.

        Returns:
            int: The frame id.
        """
        return self.frame_id

    def get_intrinsics(self) -> np.ndarray:
        """Return the camera intrinsics matrix.

        Returns:
            np.ndarray: The intrinsics matrix. (3x3)
        """
        return self.intrinsics

    def show(self, other=None) -> None:
        """Show the image of the frame."""
        # show image with frame index
        if other is None:
            cv.imshow("Frame {}".format(self.frame_id), self.image)
        else:
            # add some padding to the image
            img1 = cv.copyMakeBorder(
                other.image, 20, 20, 20, 20, cv.BORDER_CONSTANT, value=(255, 255, 255)
            )
            img2 = cv.copyMakeBorder(
                self.image, 20, 20, 20, 20, cv.BORDER_CONSTANT, value=(255, 255, 255)
            )
            # add description
            img = np.hstack((img1, img2))
            cv.imshow("Frame {} and {}".format(other.frame_id, self.frame_id), img)

        k = cv.waitKey(30) & 0xFF
        if k == 27:
            cv.destroyAllWindows()

    def show_features(self) -> None:
        """Return a copy of the image with features drawn on it.

        Returns:
            np.array: A copy of the image with features drawn on it.
        """
        # TODO: Test this method. Could not test it because we don't have the features yet.
        img = cv.drawKeypoints(
            self.image,
            self.features.keypoints,
            self.image,
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        cv.imshow("Frame {}".format(self.frame_id), img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def __repr__(self) -> str:
        return "Frame id: {}".format(self.frame_id)
