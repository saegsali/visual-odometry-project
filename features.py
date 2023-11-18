from typing import Tuple, List
import numpy as np
import cv2

from sensors import Camera


class FeatureExtractor:
    """
    A class to handle feature detection and description extraction in images.
    """
    
    def __init__(self, detector_descriptor: cv2.Feature2D):
        """
        Initialize the FeatureExtractor with a detector/descriptor object.
        
        :param detector_descriptor: An OpenCV feature detector and descriptor.
        """
        self.detector_descriptor = detector_descriptor

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detect and compute features and descriptors from the image.
        
        :param image: The image to process.
        :return: A tuple containing the list of keypoints and descriptors.
        """
        pass


class FeatureMatcher:
    """
    A class to match features between pairs of images.
    """
    
    def __init__(self, matcher_type: cv2.DescriptorMatcher):
        """
        Initialize the FeatureMatcher with a specific OpenCV DescriptorMatcher.
        
        :param matcher_type: An OpenCV DescriptorMatcher object.
        """
        self.matcher_type = matcher_type

    def match(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> List[cv2.DMatch]:
        """
        Match features between two sets of descriptors.
        
        :param descriptors1: The first set of descriptors.
        :param descriptors2: The second set of descriptors.
        :return: A list of matches.
        """
        pass

class MotionEstimator:
    """
    A class to estimate the motion between two frames based on matched features.
    """
    
    def __init__(self, method: str):
        """
        Initialize the MotionEstimator with a method for motion estimation.
        
        :param method: A string indicating the method used for motion estimation.
        """
        self.method = method

    def estimate_motion(self, matches: List[cv2.DMatch], keypoints1: List[cv2.KeyPoint], keypoints2: List[cv2.KeyPoint], camera: Camera) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate motion between two sets of keypoints.
        
        :param matches: The matches between keypoints.
        :param keypoints1: The keypoints in the first image.
        :param keypoints2: The keypoints in the second image.
        :param camera: The Camera object with intrinsic parameters.
        :return: A tuple containing the rotation and translation vectors.
        """
        pass