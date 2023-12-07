import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy import signal

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
        patch_size: int = 9,
        kappa: float = 0.08,
        num_keypoints: int = 200,
        nonmaximum_supression_radius: int = 5,
        descriptor_radius: int = 9,
        match_lambda: float = 5.0,
    ):
        """Initialize feature matcher and set parameters."""
        self._patch_size = patch_size
        self._kappa = kappa
        self._num_keypoints = num_keypoints
        self._nonmaximum_supression_radius = nonmaximum_supression_radius
        self._descriptor_radius = descriptor_radius
        self._match_lambda = match_lambda

    def to_gray(self, frame) -> Frame:
        """Converts the image to a grayscale image."""
        img = frame.image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return Frame(gray_img)

    def extractKeypoints(self, frame) -> Frame:
        """Calculates the harris scores for an image given a patch size and a kappa value
        Selects the best scores as keypoints and performs non-maximum supression of a (2r + 1)*(2r + 1) box around
        the current maximum.

        Args:
            frame (Frame): object containing an image

        Returns:
            keypoints (Frame): object containing the input image and its keypoints
        """
        pass
        # Get image from input frame
        img = frame.image

        # Init sobel filters in both directions
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Calculate the convolutions filters and image needed for the harris score
        I_x = signal.convolve2d(sobel_x, img, mode="valid", boundary="symm")
        I_y = signal.convolve2d(sobel_y, img, mode="valid", boundary="symm")

        I_x2 = I_x**2
        I_y2 = I_y**2
        I_xy = I_x * I_y

        patch = np.ones(shape=(self._patch_size, self._patch_size))
        patch_radius = self._patch_size // 2

        I_x2_sum = signal.convolve2d(patch, I_x2, mode="valid", boundary="symm")
        I_y2_sum = signal.convolve2d(patch, I_y2, mode="valid", boundary="symm")
        I_xy_sum = signal.convolve2d(patch, I_xy, mode="valid", boundary="symm")

        # Calculate the harris scores of the image of the input frame
        trace = I_x2_sum + I_y2_sum
        determinant = I_x2_sum * I_y2_sum - I_xy_sum**2

        harris_scores = determinant - self._kappa * (trace**2)
        harris_scores[harris_scores < 0] = 0

        harris_scores = np.pad(
            harris_scores,
            [
                (patch_radius + 1, patch_radius + 1),
                (patch_radius + 1, patch_radius + 1),
            ],
            mode="constant",
            constant_values=0,
        )

        # Get shape of Harris score array (image + padding)
        h, w = harris_scores.shape
        r = self._nonmaximum_supression_radius

        # Init array containing the keypoints
        kp = np.zeros([self._num_keypoints, 2, 1])

        # Get pixel cooridates of the (num_keypoints) keypoints with the highest harris scores
        # and set pixels with radius (nonmaximum_supression_radius) around keypoint to 0 to not select it again as keypoint
        for i in range(self._num_keypoints):
            h_max, w_max = np.argmax(harris_scores) // w, np.argmax(harris_scores) % w
            harris_scores[h_max - r : h_max + r + 1, w_max - r : w_max + r + 1] = 0

            kp[i, :, :] = np.array([[h_max], [w_max]])

        # Init Frame with keypoints
        kp_feature = Features(kp)
        keypoints = Frame(img, kp_feature)

        return keypoints

    def extractDescriptors(self, frame) -> Frame:
        """
        Returns a Frame with a N x (2*r+1)^2 x 1 matrix of image patch vectors describing the extracted keypoints of the Frame.
        r is the descriptor_radius.

        Args:
            frame (Frame): object containing an image and its keypoints

        Returns:
            descriptors (Frame): object containing the input image, its keypoints and descriptors
        """
        # Get image and keypoints from input frame
        img = frame.image
        keypoints = frame.features.keypoints

        r = self._descriptor_radius
        N = self._num_keypoints

        # Init descriptor matrix and padded image
        desc = np.zeros([N, (2 * r + 1) ** 2, 1])
        padded = np.pad(img, [(r, r), (r, r)], mode="constant", constant_values=0)

        # For all keypoints, extract the descriptor patch around it
        for kp in range(N):
            patch = padded[
                int(keypoints[kp, 0, 0]) - r + r : int(keypoints[kp, 0, 0]) + r + 1 + r,
                int(keypoints[kp, 1, 0]) - r + r : int(keypoints[kp, 1, 0]) + r + 1 + r,
            ]
            desc[kp, :, :] = patch.reshape(-1, 1)

        # Init Frame with keypoints and descriptors
        desc_feature = Features(keypoints, desc)
        descriptors = Frame(img, desc_feature)

        return descriptors

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
        num_desc1 = descriptors1.shape[0]
        num_desc2 = descriptors2.shape[0]

        # compute euclidean distance of every descriptor pair
        dists = cdist(
            descriptors1.reshape([num_desc1, -1]),
            descriptors2.reshape([num_desc2, -1]),
            "euclidean",
        )
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
        matches_pairs = np.hstack(
            (row_indices.reshape(-1, 1), unique_matches.reshape(-1, 1))
        )

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
    sequence = Sequence("parking", camera=1, use_lowres=True)
    frame1 = sequence.get_frame(0)
    frame2 = sequence.get_frame(6)

    # Create an instance of the HarrisCornerDetector class
    harris = HarrisCornerDetector(match_lambda=5, nonmaximum_supression_radius=5)

    frame1 = harris.to_gray(frame1)
    frame1 = harris.extractKeypoints(frame1)
    frame1 = harris.extractDescriptors(frame1)

    frame2 = harris.to_gray(frame2)
    frame2 = harris.extractKeypoints(frame2)
    frame2 = harris.extractDescriptors(frame2)

    # # Compute matches using SIFT (TODO: replace later with our FeatureMatcher)
    # sift = cv2.SIFT_create()
    # keypoints1, descriptors1 = sift.detectAndCompute(frame1.image, None)
    # keypoints2, descriptors2 = sift.detectAndCompute(frame2.image, None)

    # keypoints1 = np.array([kp.pt for kp in keypoints1]).reshape(-1, 2, 1)
    # keypoints2 = np.array([kp.pt for kp in keypoints2]).reshape(-1, 2, 1)
    # descriptors1 = np.array(descriptors1)
    # descriptors2 = np.array(descriptors2)

    # frame1.features = Features(keypoints1, descriptors1)
    # frame2.features = Features(keypoints2, descriptors2)

    # Call the matchDescriptor method
    matches = harris.matchDescriptor(frame1, frame2)
    print('number of matches:', matches.frame1.features.keypoints.shape[0])

    # Visualize
    img1 = matches.frame1.image
    img2 = matches.frame2.image
    keypoint1 = matches.frame1.features.keypoints
    keypoint2 = matches.frame2.features.keypoints

    def draw_keypoints(img, keypoints):
        img = img.copy()
        H, W = img.shape[:2]

        for i, keypoint in enumerate(keypoints):
            # print(keypoint)
            x1 = keypoint[0].astype(int)
            y1 = keypoint[1].astype(int)
            # print(x1.reshape(-1).shape)
            img = cv2.circle(
                img,
                (x1[0], y1[0]),
                radius=2,
                color=((10 * i) % 255, i % 255, 255),
                thickness=2,
            )
            # print(i)
        return img

    img1 = draw_keypoints(img1, np.flip(keypoint1, axis=1))
    img2 = draw_keypoints(img2, np.flip(keypoint2, axis=1))

    img_both = cv2.hconcat([img1, img2])

    cv2.imshow("Keypoints", img_both)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
