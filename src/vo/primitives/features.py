import numpy as np


class Features:
    def __init__(
        self,
        keypoints: np.ndarray,
        landmarks: np.ndarray = None,
        uids: np.ndarray = None,
    ) -> None:
        """Initializes a features object.

        Args:
            keypoints (np.ndarray): array of keypoints pixel locations, shape = (N, 2).
            descriptors (np.ndarray, optional): array of keypoints descriptors, shape = (N, D). Defaults to None.
        """
        super().__init__()
        n_keypoints = keypoints.shape[0]
        self._keypoints = keypoints

        # Use setter methods for all other properties to ensure consistency
        self.descriptors = None

        self.landmarks = (
            landmarks
            if landmarks is not None
            else np.nan * np.ones(shape=(n_keypoints, 3, 1))
        )  # initialize landmarks as unkown
        self.state = np.zeros(
            shape=(n_keypoints,)
        )  # 0: unmatched, 1: matched, 2: triangulated

        self._uids = uids

        # Instantiate inliers arrays (at start all inliers by default)
        self._p3p_inliers = np.ones(shape=(self.length)).astype(bool)
        self._triangulate_inliers = np.ones(shape=(self.length)).astype(bool)

    @property
    def matched_inliers_keypoints(self) -> np.ndarray:
        return self._keypoints[self.match_inliers]

    @property
    def triangulate_inliers_keypoints(self) -> np.ndarray:
        return self._keypoints[self._triangulate_inliers]

    @property
    def p3p_inliers_keypoints(self) -> np.ndarray:
        return self._keypoints[self._p3p_inliers]

    @property
    def match_inliers(self) -> np.ndarray:
        return self.state >= 1

    @property
    def keypoints(self) -> np.ndarray:
        return self._keypoints

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def descriptors(self) -> np.ndarray:
        return self._descriptors

    @property
    def landmarks(self) -> np.ndarray:
        return self._landmarks

    @property
    def inliers(self) -> np.ndarray:
        return self._inliers

    @property
    def uids(self) -> np.ndarray:
        return self._uids

    @keypoints.setter
    def keypoints(self, keypoints: np.ndarray) -> None:
        """Set keypoints and check if number of keypoints and descriptors are equal.

        Args:
            keypoints (np.ndarray): _description_
        """
        assert (
            keypoints.shape[0] == self._keypoints.shape[0]
        ), "Unequal number of keypoints."
        self._keypoints = keypoints

    @state.setter
    def state(self, state: np.ndarray) -> None:
        """Set state of keypoints.

        Args:
            state (np.ndarray): _description_
        """
        assert (
            state.shape[0] == self._keypoints.shape[0]
        ), "Unequal number of state and keypoints."
        self._state = state

    @descriptors.setter
    def descriptors(self, descriptors: np.ndarray) -> None:
        """Set descriptors and check if number of descriptors and keypoints are equal.

        Args:
            descriptors (np.ndarray): _description_
        """
        assert (
            descriptors is None or descriptors.shape[0] == self._keypoints.shape[0]
        ), "Unequal number of descriptors and keypoints."
        self._descriptors = descriptors

    @landmarks.setter
    def landmarks(self, landmarks: np.ndarray) -> None:
        assert (
            landmarks.shape[0] == self._keypoints.shape[0]
        ), "Unequal number of landmarks and keypoints."
        self._landmarks = landmarks

    @uids.setter
    def uids(self, uids: np.ndarray) -> None:
        assert (
            uids is None or uids.shape[0] == self._keypoints.shape[0]
        ), "Unequal number of uids and keypoints."
        self._uids = uids

    @property
    def length(self) -> int:
        assert (
            self._descriptors is None
            or self._descriptors.shape[0] == self._keypoints.shape[0]
        ), "Unequal number of descriptors and keypoints."
        assert (
            self._landmarks is None
            or self._landmarks.shape[0] == self._keypoints.shape[0]
        ), "Unequal number of landmarks and keypoints."
        assert (
            self._state.shape[0] == self._keypoints.shape[0]
        ), "Unequal number of state and keypoints."

        return self._keypoints.shape[0]

    def set_p3p_inliers(self, inliers: np.ndarray) -> None:
        """Updates the inliers mask.

        Args:
            inliers (np.ndarray, optional): An inliers mask to apply to current inlier mask.
        """
        self._p3p_inliers = np.zeros_like(self._p3p_inliers)
        self._p3p_inliers[(self._state >= 1) & self._triangulate_inliers] = inliers

    def set_triangulate_inliers(self, inliers: np.ndarray) -> None:
        """Updates the inliers mask.

        Args:
            inliers (np.ndarray, optional): An inliers mask to apply to current inlier mask.
        """
        self._triangulate_inliers = np.zeros_like(self._triangulate_inliers)
        self._triangulate_inliers[self._state >= 1] = inliers
