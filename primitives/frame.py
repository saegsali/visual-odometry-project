import numpy as np
import cv2

from sensors import Camera
from primitives import Features


class Frame:
    """A class to represent a single frame in a video sequence."""
    
    def __init__(self, image: np.ndarray, features: Features=None, sensor: Camera=None):
        self.image = image
        self.frame_id = None
        self.features = features
        self.sensor = sensor

    def show(self):
        """Show the image of the frame."""
        # show image with frame index
        cv2.imshow("Frame {}".format(self.frame_id), self.image)

    def draw_features(self):
        """Return a copy of the image with features drawn on it.

        Returns:
            np.array: A copy of the image with features drawn on it.
        """
        # TODO: draw features on image
        Exception("Not Implemented Error")
        return self.image
    
    
    
