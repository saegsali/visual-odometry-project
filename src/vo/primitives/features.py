import numpy as np


class Features:
    def __init__(
        self,
        keypoints: np.ndarray,
        descriptors: np.ndarray = None,
        landmarks: np.ndarray = None,
        uids: np.ndarray = None,
        inliers: np.ndarray = None,
    ) -> None:
        """Initializes a features object.

        Args:
            keypoints (np.ndarray): array of keypoints pixel locations, shape = (N, 2).
            descriptors (np.ndarray, optional): array of keypoints descriptors, shape = (N, D). Defaults to None.
            landmarks (np.ndarray, optional): array of associated landmark 3D position in world coordinates, shape = (N, 3). Defaults to None.
        """
        super().__init__()
        self._keypoints = keypoints
        self._descriptors = descriptors
        self._landmarks = landmarks

        self._uids = uids
        self._inliers = (
            np.ones(shape=(self.length)).astype(bool)
            if inliers is None
            else inliers.astype(bool)
        )

    @property
    def keypoints(self) -> np.ndarray:
        return self._keypoints

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

    @descriptors.setter
    def descriptors(self, descriptors: np.ndarray) -> None:
        """Set descriptors and check if number of descriptors and keypoints are equal.

        Args:
            descriptors (np.ndarray): _description_
        """
        assert self._descriptors is None, "Descriptors already set."
        assert (
            descriptors.shape[0] == self._keypoints.shape[0]
        ), "Unequal number of descriptors and keypoints."
        self._descriptors = descriptors

    @landmarks.setter
    def landmarks(self, landmarks: np.ndarray) -> None:
        assert self._landmarks is None, "Landmarks already set."
        assert (
            landmarks.shape[0] == self._keypoints.shape[0]
        ), "Unequal number of landmarks and keypoints."
        self._landmarks = landmarks

    @inliers.setter
    def inliers(self, inliers: np.ndarray) -> None:
        assert (
            inliers.shape[0] == self._keypoints.shape[0]
        ), "Unequal number of inliers and keypoints."
        self._inliers = inliers.astype(bool)

    @uids.setter
    def uids(self, uids: np.ndarray) -> None:
        assert (
            uids.shape[0] == self._keypoints.shape[0]
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

        return self._keypoints.shape[0]

    def apply_inliers(self, inliers: np.ndarray = None) -> None:
        """Applies the inliers mask to keypoints, descriptors and landmarks and removes outliers from arrays.

        Args:
            inliers (np.ndarray, optional): An updated inliers mask to use, otherwise uses stored inliers. Defaults to None.
        """
        if inliers is not None:
            self.inliers = inliers

        self._keypoints = self._keypoints[self._inliers]
        if self._descriptors is not None:
            self._descriptors = self._descriptors[self._inliers]
        if self._landmarks is not None:
            self._landmarks = self._landmarks[self._inliers]

        self._inliers = np.ones(shape=(self.length)).astype(bool)
