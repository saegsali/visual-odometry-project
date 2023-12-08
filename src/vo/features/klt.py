import numpy as np
import time
import cv2
import sys

from vo.primitives import Sequence, Frame, Features
from vo.visualization.overlays import display_fps


class KLTTracker:
    """A class to track features using the KLT algorithm.

    Args:
        frame (Frame): The frame to track features in.

    Methods:
        track_features: Track features in the current frame using the KLT algorithm.
        draw_tracks: Draw the tracks of the last 5 frames on the image.

    """

    # params for (ShiTomasi) corner detection
    _feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas Kanade optical flow
    _lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # Create some random colors
    _colors = np.random.randint(0, 255, (_feature_params["maxCorners"], 3))

    # Only keep points with error less than this threshold
    _error_threshold = 30

    def __init__(self, frame):
        self._min_inliers = 90
        self._num_features = None
        self._old_frame = frame
        self._frame = frame
        self._frame.features = Features(
            keypoints=self.find_corners(frame=self._frame)
        )  # Initialize features
        self._frame.features.uids = self._get_udis(self._frame.features.length)
        self._last_masks = []  # List to store the last 5 masks for drawing tracks

    @property
    def frame(self) -> Frame:
        return self._frame

    @property
    def old_img_gray(self) -> np.ndarray:
        return cv2.cvtColor(self._old_frame.image, cv2.COLOR_BGR2GRAY)

    @property
    def img_gray(self) -> np.ndarray:
        return cv2.cvtColor(self._frame.image, cv2.COLOR_BGR2GRAY)

    @frame.setter
    def frame(self, frame: Frame) -> None:
        self._frame = frame
        self._frame.features = Features(self.find_corners(self._frame))

    def _get_udis(self, length: int) -> np.ndarray:
        """Generate a random array of integers."""
        return np.random.randint(0, np.iinfo(np.int32).max, size=length, dtype=np.int32)

    def _fill_udis(self, array: np.ndarray, target_length: int) -> np.ndarray:
        """Fill an array with random integers to a target length."""
        return np.concatenate((array, self._get_udis(target_length - array.shape[0])))

    def to_gray(self, img) -> np.ndarray:
        """Converts the image to a grayscale image."""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def find_corners(self, frame: Frame, mask=None) -> np.ndarray:
        # Ensure the input image is grayscale
        img = frame.image
        if len(img.shape) == 3:
            img = self.to_gray(img)
        features = cv2.goodFeaturesToTrack(img, mask=mask, **self._feature_params)
        self._num_features = features.shape[0]
        return features

    def track_features(self, frame: Frame) -> Frame:
        """Track features in the current frame using the KLT algorithm. Freatures are updated if there are not enough inliers.

        Args:
            frame (Frame): The current frame.

        Returns:
            Frame: The current frame with updated features.
        """

        # Update the frame
        self._old_frame = self._frame
        self._frame = frame

        # Extract inliers from previous frame
        prev_pts = self._old_frame.features.keypoints

        # Calculate optical flow using the KLT algorithm
        next_pts, status, error = cv2.calcOpticalFlowPyrLK(
            prevImg=self.old_img_gray,
            nextImg=self.img_gray,
            prevPts=prev_pts,
            nextPts=None,
            **self._lk_params,
        )

        # Filter features based on status
        inliers = status.flatten().astype(bool)
        # create a maks to remove points with high error
        error = (error < self._error_threshold).flatten().astype(bool)

        # logical and of the two masks
        filter = np.logical_and(inliers, error)

        # remove points which are not inliers and have high error
        next_pts = next_pts[filter]

        # Update keypoints in frame
        uids = self._old_frame.features.uids[filter]
        uids = self._fill_udis(uids, next_pts.shape[0])

        self.frame.features = Features(keypoints=next_pts, uids=uids)
        self._old_frame.features = Features(keypoints=prev_pts[filter], uids=uids)

        # Update template features if there are not enough inliers in the current frame
        if (
            np.count_nonzero(filter) < self._min_inliers
            or np.count_nonzero(filter)
            < self._num_features
            * 0.8  # less than 80% of the initial features are inliers
        ):
            if sys.gettrace() is not None:
                print("Adding new features")
            mask = np.ones_like(self.img_gray) * 255
            # loop over the keypoints (N,1,2) and draw them on the mask
            for x, y in self.frame.features.keypoints.reshape(-1, 2).astype(int):
                cv2.circle(mask, (x, y), 5, 0, -1)
            # Update template features
            new_features = Features(self.find_corners(frame=self.frame, mask=mask))
            self.frame.features = Features(
                keypoints=np.concatenate(
                    (
                        self.frame.features.keypoints[self.frame.features.inliers],
                        new_features.keypoints,
                    )
                )
            )
            # Update uidses of the new features
            self.frame.features.uids = self._fill_udis(
                self._old_frame.features.uids, self.frame.features.length
            )
            # Reset the mask for drawing tracks
            # self._last_masks = [] # Uncomment this line to reset the mask
        print(
            "Number of features:",
            len(self.frame.features.keypoints[self.frame.features.inliers]),
        )
        return self.frame

    def draw_tracks(self) -> np.ndarray:
        """Draw the tracks of the last 5 frames on the image.

        Returns:
            np.ndarray: The image with the tracks drawn on it.
            np.ndarray: The mask with the tracks drawn on it.
        """

        img = self.frame.image.copy()
        mask = np.zeros_like(img)

        for i, (new, old) in enumerate(
            zip(
                self.frame.features.keypoints[self.frame.features.inliers],
                self._old_frame.features.keypoints[self._old_frame.features.inliers],
            )
        ):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(
                mask,
                (int(a), int(b)),
                (int(c), int(d)),
                self._colors[self.frame.features.uids[i] % len(self._colors)].tolist(),
                2,
            )
            img = cv2.circle(
                img,
                (int(a), int(b)),
                5,
                self._colors[self.frame.features.uids[i] % len(self._colors)].tolist(),
                -1,
            )
        # Update last masks
        self._update_last_masks(mask)

        # Overlay the last 5 masks on the image
        img = self._overlay_last_masks(img)
        return img, mask

    def _update_last_masks(self, mask):
        # Maintain only the last 5 masks
        self._last_masks.append(mask)
        if len(self._last_masks) > 10:
            self._last_masks = self._last_masks[-10:]

    def _overlay_last_masks(self, img):
        # Overlay the last 5 masks on the image
        overlay = np.zeros_like(img)
        for mask in self._last_masks:
            overlay = cv2.add(overlay, mask)
        return cv2.addWeighted(img, 1, overlay, 0.5, 0)


# Example usage:
if __name__ == "__main__":
    # Create a video object
    video = Sequence("kitti", "data", camera=0, increment=1)
    # Get the first frame
    frame = video.get_frame(0)

    # Initialize the tracker with the first frame
    klt_tracker = KLTTracker(frame=frame)

    # Variables for calculating FPS
    start_time = time.time()

    for i in range(1, len(video)):
        frame = next(video)

        # Calculate optical flow
        frame = klt_tracker.track_features(frame)

        # Draw the tracks on the image
        image, mask = klt_tracker.draw_tracks()

        # Overlay the FPS on the image
        image = display_fps(image=image, start_time=start_time, frame_count=i)

        # Display the resulting frame
        cv2.imshow("Press esc to stop", image)

        k = cv2.waitKey(30) & 0xFF  # 30ms delay -> try lower value for more FPS :)
        if k == 27:
            break

    cv2.destroyAllWindows()
