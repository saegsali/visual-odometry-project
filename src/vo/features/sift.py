import cv2
import numpy as np

from vo.primitives import Features, Frame, Matches


class SIFTDetector:
    def __init__(self, frame: Frame = None):
        """Initialize feature matcher and set parameters."""
        self._frame1 = frame
        self._frame2 = frame

    def get_sift_matches(self, frame) -> Matches:
        """Track features of 2 frames using the SIFT detector algorithm.

        Args:
            frame (Frame): The new frame.

        Returns:
            matches (Matches): object containing the matching freature points of the input frames.
        """

        # Update the frame
        self._frame1 = self._frame2
        self._frame2 = frame

        sift = cv2.SIFT_create()
        keypoints1_raw, descriptors1 = sift.detectAndCompute(self._frame1.image, None)
        keypoints1 = np.array([kp.pt for kp in keypoints1_raw]).reshape(-1, 2, 1)
        descriptors1 = np.array(descriptors1)
        landmarks1 = None

        keypoints2_raw, descriptors2 = sift.detectAndCompute(self._frame2.image, None)
        keypoints2 = np.array([kp.pt for kp in keypoints2_raw]).reshape(-1, 2, 1)
        descriptors2 = np.array(descriptors2)

        self._frame1.features = Features(keypoints1, descriptors1, landmarks=landmarks1)
        self._frame2.features = Features(keypoints2, descriptors2)

        # Match keypoints
        index_params = dict(algorithm=0, trees=20)
        search_params = dict(checks=150)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # # Need to draw only good matches, so create a mask
        # good_matches = [[0, 0] for i in range(len(matches))]

        # # Good matches
        # for i, (m, n) in enumerate(matches):
        #     if m.distance < 0.5 * n.distance:
        #         good_matches[i] = [1, 0]

        # # plot matches
        # img = cv2.drawMatchesKnn(
        #     frame1.image,
        #     keypoints1_raw,
        #     frame2.image,
        #     keypoints2_raw,
        #     matches,
        #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        # )
        # cv2.imshow("Press esc to stop", img)
        # k = cv2.waitKey(30) & 0xFF  # 30ms delay -> try lower value for more FPS :)
        # if k == 27:
        #     break

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append([m.queryIdx, m.trainIdx])
        good = np.array(good)

        # Visualize fundamental matrix
        matches = Matches(self._frame1, self._frame2, matches=good)

        return matches
