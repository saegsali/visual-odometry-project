import cv2
import numpy as np

from vo.primitives import Features, Frame, Matches


class SIFTDetector:
    def __init__(self, frame: Frame):
        """Initialize feature matcher and set parameters."""
        self.sift = cv2.SIFT_create()

        kp, desc = self.detect_and_compute(frame=frame)
        frame.features = Features(keypoints=kp)
        frame.features.descriptors = desc

    def detect_and_compute(self, frame: Frame) -> tuple[np.ndarray]:
        keypoints1_raw, descriptors1 = self.sift.detectAndCompute(frame.image, None)
        keypoints1 = np.array([kp.pt for kp in keypoints1_raw]).reshape(-1, 2, 1)
        descriptors1 = np.array(descriptors1)

        return keypoints1, descriptors1

    def get_sift_matches(self, curr_frame: Frame, new_frame: Frame) -> Matches:
        """Track features of 2 frames using the SIFT detector algorithm.

        Args:
            frame (Frame): The new frame.

        Returns:
            matches (Matches): object containing the matching freature points of the input frames.
        """
        # Detect keypoints in new frame
        kp2, desc2 = self.detect_and_compute(new_frame)

        new_frame.features = Features(kp2)
        new_frame.features.descriptors = desc2

        # Match keypoints
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(
            curr_frame.features.descriptors, new_frame.features.descriptors, k=2
        )

        # Apply ratio test
        good = []
        used = np.zeros(shape=(len(new_frame.features.descriptors)))

        for m, n in matches:
            assert m.distance < n.distance
            if m.distance < 0.8 * n.distance:
                if used[m.trainIdx] == 0:
                    good.append([m.queryIdx, m.trainIdx])
                    used[m.trainIdx] = 1
        good = np.array(good)
        matches = Matches(curr_frame, new_frame, matches=good)

        return matches
