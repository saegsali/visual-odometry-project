import numpy as np
import time
import cv2
import sys

from vo.primitives import Sequence, Frame, Features
from vo.visualization.overlays import display_fps
from vo.primitives import Matches


class KLTTracker:
    """A class to track features using the KLT algorithm.

    Args:
        frame (Frame): The frame to track features in.

    Methods:
        track_features: Track features in the current frame using the KLT algorithm.
        draw_tracks: Draw the tracks of the last 5 frames on the image.

    """

    # params for (ShiTomasi) corner detection
    _feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=7, blockSize=7)

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
        # If array is None, return an array of random integers
        if array is None:
            return self._get_udis(target_length)
        else:
            return np.concatenate(
                (array, self._get_udis(target_length - array.shape[0]))
            )

    def to_gray(self, img) -> np.ndarray:
        """Converts the image to a grayscale image."""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def find_corners(
        self, frame: Frame, mask=None, use_goodFeaturesToTrack=True
    ) -> np.ndarray:
        # Ensure the input image is grayscale
        img = frame.image
        if len(img.shape) == 3:
            img = self.to_gray(img)

        if (
            use_goodFeaturesToTrack
        ):  # use opencv's goodFeaturesToTrack (default is ShiTomasi)
            features = cv2.goodFeaturesToTrack(img, mask=mask, **self._feature_params)
        else:  # use Harris corner detection with subpixel refinement
            dst = cv2.cornerHarris(img, 2, 3, 0.04)
            dst = cv2.dilate(dst, None)
            ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
            dst = np.uint8(dst)
            # find centroids
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
            # define the criteria to stop and refine the corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            features = cv2.cornerSubPix(
                img, np.float32(centroids), (5, 5), (-1, -1), criteria
            )

        # reshape features to (N,2,1)
        features = features.reshape((-1, 2, 1))
        self._num_features = features.shape[0]
        return features

    def update_features(self, new_keypoints: np.ndarray) -> Features:
        """Update the keypoints of the input frame. The keypoints are updated with the new keypoints.
        Landmarks are extended with np.nan and the state is extended with 0. The uids are extended with random integers.
        The tracks are extended with the new keypoints. The poses are extended with np.eye(4) and the candidate_mask is extended with False.

        Args:
            new_keypoints (np.ndarray): The new keypoints.

        Returns:
            Features: The updated features.
        """
        features = self._old_frame.features

        # Update keypoints
        keypoints = np.concatenate((features.keypoints, new_keypoints))
        # Extend landmarks (N,3,1) with np.nan
        landmarks = np.concatenate(
            (
                features.landmarks if features.landmarks is not None else np.array([]),
                np.nan * np.ones(shape=(new_keypoints.shape[0], 3, 1)),
            )
        )
        # Extend state (N,1) with 0
        state = np.concatenate(
            (
                features.state if features.state is not None else np.array([]),
                np.zeros(new_keypoints.shape[0]),  # Create a 1D array of zeros
            )
        )
        # Extend uids (N,1) with random integers
        uids = self._fill_udis(features.uids, new_keypoints.shape[0] + features.length)
        # Extend tracks (N,2,1) with new keypoints
        tracks = np.concatenate((features.tracks, new_keypoints))
        # Extend poses (N,4,4) with np.eye(4)
        poses = np.concatenate(
            (
                features.poses if features.poses is not None else np.array([]),
                np.stack([np.eye(4)] * new_keypoints.shape[0]),
            )
        )
        # Extend candidate_mask (N,1) with False
        candidate_mask = np.concatenate(
            (
                features.candidate_mask
                if features.candidate_mask is not None
                else np.array([]),
                np.zeros(keypoints.shape[0] - features.candidate_mask.shape[0]).astype(
                    bool
                ),
            )
        )
        assert (
            keypoints.shape[0]
            == landmarks.shape[0]
            == state.shape[0]
            == uids.shape[0]
            == tracks.shape[0]
            == poses.shape[0]
            == candidate_mask.shape[0]
        ), "The number of keypoints, landmarks, state, uids, tracks, poses and candidate_mask must be the same."

        # Initialize features
        features = Features(
            keypoints=keypoints,
            landmarks=landmarks,
        )
        # Update feature properties
        features.state = state
        features.uids = uids
        features.tracks = tracks
        features.poses = poses
        features.candidate_mask = candidate_mask
        return features

    def track_features(self, curr_frame: Frame, new_frame: Frame) -> Matches:
        """Track features in the current frame using the KLT algorithm. Freatures are updated if there are not enough inliers.

        Args:
            curr_frame (Frame): The current frame.
            new_frame (Frame): The new frame.

        Returns:
            matches (Matches): object containing the matching freature points of the input frame and the one before.
        """

        # Update the frame
        self._old_frame = curr_frame
        self._frame = new_frame

        # Update template features if there are not enough inliers in the current frame
        if (
            self._old_frame.features is None
            # or self._old_frame.features.length < self._num_features
            or self._old_frame.features.length
            < self._num_features
            * 0.8  # less than 80% of the initial features are inliers
        ):
            if sys.gettrace() is not None:
                print("Adding new features")
            mask = np.ones_like(self.img_gray) * 255
            if self._old_frame.features is None:  # first frame has no features
                # loop over the keypoints (N,1,2) and draw them on the mask
                for x, y in self._old_frame.features.keypoints.reshape(-1, 2).astype(
                    int
                ):
                    cv2.circle(mask, (x, y), 3, 0, -1)
            # Update template features
            new_keypoints = self.find_corners(frame=self._old_frame, mask=mask)

            self._old_frame.features = self.update_features(new_keypoints=new_keypoints)
            # Update uidses of the new features
            self._old_frame.features.uids = self._fill_udis(
                self._old_frame.features.uids, self._old_frame.features.length
            )

        # Calculate optical flow using the KLT algorithm
        next_pts, status, error = cv2.calcOpticalFlowPyrLK(
            prevImg=self.old_img_gray,
            nextImg=self.img_gray,
            prevPts=self._old_frame.features.keypoints,
            nextPts=None,
            **self._lk_params,
        )
        # reshape next_pts to (N,2,1)
        next_pts = next_pts.reshape((-1, 2, 1))

        # Filter features based on status
        inliers = status.flatten().astype(bool)
        # create a maks to remove points with high error
        error = (error < self._error_threshold).flatten().astype(bool)

        # logical and of the two masks
        filter = np.logical_and(inliers, error)

        # # remove points which are not inliers and have high error
        # next_pts = next_pts[filter]

        self.frame.features = Features(
            keypoints=next_pts
        )  # Initialize features for the current frame

        # Update keypoints in frame
        uids = self._old_frame.features.uids
        uids = self._fill_udis(uids, next_pts.shape[0])
        self.frame.features.uids = uids

        # Mask the features to remove the outliers
        self.frame.features.mask(filter)
        self._old_frame.features.mask(filter)

        # Create numpy array indicating the matching keypoint of the other frame (here same index correspond to same keypoint)
        matches_index = np.arange(0, self.frame.features.keypoints.shape[0])
        matches_pairs = np.hstack(
            (matches_index.reshape(-1, 1), matches_index.reshape(-1, 1))
        )

        # Create Matches object with the current and the old frame
        self._matches = Matches(self._old_frame, self.frame, matches_pairs)

        return self._matches

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
                self.frame.features.keypoints,
                self._old_frame.features.keypoints,
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
        curr_frame = frame
        frame = next(video)

        # Calculate optical flow
        matches = klt_tracker.track_features(curr_frame, frame)

        # Draw the tracks on the image
        image, mask = klt_tracker.draw_tracks()

        # Display the resulting frame
        cv2.imshow("Press esc to stop", image)

        k = cv2.waitKey(30) & 0xFF  # 30ms delay -> try lower value for more FPS :)
        if k == 27:
            break

    cv2.destroyAllWindows()
