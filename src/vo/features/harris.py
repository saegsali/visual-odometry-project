import cv2
import numpy as np
from scipy.spatial.distance import cdist

from vo.helpers import (
    to_cartesian_coordinates,
    to_homogeneous_coordinates,
    normalize_points,
)
from vo.primitives import Features, Frame, Matches
from vo.algorithms import RANSAC


class HarrisCornerDetector:
    def __init__(
        self,
        match_lambda: float = 4.0,
    ):
        """Initialize feature matcher and set parameters."""
        self._match_lambda = match_lambda

    def extractFeatures(self):
        pass
        #TODO

    def matchDescriptor(self, frame1, frame2) -> Matches:
        """Matches the features of two frames based on their descriptors.

        Args:
            frame1 (Frame): object containing the features of the first frame.
            frame2 (Frame): object containing the features of the second frame.

        Returns:
            matches (Matches): object containing the matching freature points of the frames.
        """
        # descriptors should have shape (nr_of_keypoints, size_of_descriptor)
        descriptors1 = frame1.features.descriptors
        descriptors2 = frame2.features.descriptors

        # compute euclidean distance of every descriptor pair
        dists = cdist(descriptors1, descriptors2, "euclidean")
        # dists = np.linalg.norm(descriptors1, descriptors2)
        min_non_zero_dist = np.min(dists)
        threshold = self._match_lambda * min_non_zero_dist

        # look for best matches (min distance) and set to -1 if dist > threshold
        best_matches = np.argmin(dists, axis=1)

        dists = dists[np.arange(best_matches.shape[0]), best_matches]
        best_matches[dists >= threshold] = -1

        # remove double matches
        unique_matches = np.ones_like(best_matches) * -1
        _, unique_match_idxs = np.unique(best_matches, return_index=True)
        unique_matches[unique_match_idxs] = best_matches[unique_match_idxs]

        # get the row indices and stack them horizontally with the matches
        row_indices = np.indices(unique_matches.shape)
        matches_pairs = np.hstack((row_indices.reshape(-1, 1), unique_matches.reshape(-1, 1)))

        # create a boolean mask for rows where any number is smaller than 0
        mask = (matches_pairs < 0).any(axis=1)

        # use the mask to filter out rows with entries <0
        matches_pairs_filtered = matches_pairs[~mask]

        # create and return Matches object with the extracted matches
        matches = Matches(frame1, frame2, matches_pairs_filtered)

        return matches


# Example to visualze matches
if __name__ == "__main__":
    from vo.primitives.loader import Sequence

    # Load sequence
    sequence = Sequence("malaga", camera=1, use_lowres=True)
    frame1 = sequence.get_frame(0)
    frame2 = sequence.get_frame(1)

    # Compute matches using SIFT (TODO: replace later with our FeatureMatcher)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(frame1.image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(frame2.image, None)

    keypoints1 = np.array([kp.pt for kp in keypoints1]).reshape(-1, 2, 1)
    keypoints2 = np.array([kp.pt for kp in keypoints2]).reshape(-1, 2, 1)
    descriptors1 = np.array(descriptors1)
    descriptors2 = np.array(descriptors2)

    frame1.features = Features(keypoints1, descriptors1)
    frame2.features = Features(keypoints2, descriptors2)

    # Create an instance of the HarrisCornerDetector class
    harris = HarrisCornerDetector(match_lambda=4.0) 
    # Call the matchDescriptor method
    matches = harris.matchDescriptor(frame1, frame2) 

    # Visualize
    img1 = matches.frame1.image
    img2 = matches.frame2.image
    keypoint1 = matches.frame1.features.keypoints
    keypoint2 = matches.frame2.features.keypoints

    def draw_keypoints(img, keypoints):
        img = img.copy()
        H, W = img.shape[:2]

        for i, keypoint in enumerate(keypoints):
            print(keypoint)
            x1 = keypoint[0].astype(int)
            y1 = keypoint[1].astype(int)
            print(x1.reshape(-1).shape)
            img = cv2.circle(img, (x1[0],y1[0]), radius=2, color=((10*i)%255,i%255,255), thickness=2)
        return img

    img1 = draw_keypoints(img1, keypoint1)
    img2 = draw_keypoints(img2, keypoint2)

    img_both = cv2.hconcat([img1, img2])

    cv2.imshow("Keypoints", img_both)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
