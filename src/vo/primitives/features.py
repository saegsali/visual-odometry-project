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

    @property
    def matched_candidate_inliers_keypoints(self) -> np.ndarray:
        return self._keypoints[self.matched_candidate_inliers]

    @property
    def matched_inliers_keypoints(self) -> np.ndarray:
        return self._keypoints[self.match_inliers]

    @property
    def triangulated_inliers_keypoints(self) -> np.ndarray:
        return self._keypoints[self.triangulate_inliers]

    @property
    def triangulated_inliers_landmarks(self) -> np.ndarray:
        return self._landmarks[self.triangulate_inliers]

    @property
    def p3p_inliers_keypoints(self) -> np.ndarray:
        return self._keypoints[self.p3p_inliers]

    @property
    def matched_candidate_inliers(self) -> np.ndarray:
        return self.state == 1

    @property
    def match_inliers(self) -> np.ndarray:
        return self.state >= 1

    @property
    def triangulate_inliers(self) -> np.ndarray:
        return self.state >= 2

    @property
    def p3p_inliers(self) -> np.ndarray:
        return self.state >= 2

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
