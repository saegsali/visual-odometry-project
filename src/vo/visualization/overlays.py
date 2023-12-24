import time
import cv2
import numpy as np
from vo.primitives import Features, Matches


# Function to calculate and display FPS
def display_fps(image: np.array, start_time: float, fps_queue) -> np.array:
    """Display the FPS on the image.

    Args:
        image (np.array): The image to display the FPS on.
        start_time (float): The time the processing started.
        fps_queue: queue that stores the last [maxlen] fps.

    Returns:
        np.array: The image with the FPS overlay.
        fps_queue: queue that stores the last [maxlen] fps.
    """

    current_time = time.time()
    elapsed_time = current_time - start_time
    fps = 1 / elapsed_time

    fps_queue.append(fps)
    average_fps = sum(fps_queue) / len(fps_queue)

    cv2.putText(
        image,
        f"FPS: {average_fps:.2f}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image, fps_queue


def display_keypoints_info(image: np.array, features: Features) -> np.array:
    """Display the keypoint count of the image.

    Args:
        image (np.array): The image to display the FPS on.
        features (float): The features object containing the keypoints.

    Returns:
        np.array: The image with the keypoint properties overlay.
    """
    n_keypoints = features.length
    n_matched = len(features.matched_candidate_inliers_keypoints)
    n_triangulated = len(features.triangulated_inliers_keypoints)

    cv2.putText(
        image,
        f"Keypoints: {n_keypoints}, Matched: {n_matched}, Triangulated: {n_triangulated}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def draw_keypoints(img, keypoints, colors):
    H, W = img.shape[:2]
    if isinstance(colors, tuple):
        colors = [colors] * len(keypoints)

    for keypoint, color in zip(keypoints.reshape(-1, 2), colors):
        keypoint = keypoint.astype(int)
        img = cv2.circle(
            img,
            keypoint,
            radius=2,
            color=color,
            thickness=2,
        )
    return img


def draw_lines(img, start_points, end_points, colors):
    if isinstance(colors, tuple):
        colors = [colors] * len(start_points)

    for p1, p2, color in zip(
        start_points.reshape(-1, 2), end_points.reshape(-1, 2), colors
    ):
        p1 = p1.astype(int)
        p2 = p2.astype(int)
        img = cv2.line(
            img,
            p1,
            p2,
            color=color,
            thickness=2,
        )
    return img


def plot_matches(matches: Matches) -> np.array:
    """Display the matches.

    Args:
        image (np.array): image to draw
        matches (Matches): Matches object containing current and previous frame.

    Returns:
        np.array: Image containing matches
    """
    H, W = matches.frame1.image.shape[:2]

    img1 = matches.frame1.image.copy()
    img2 = matches.frame2.image.copy()
    kp1 = matches.frame1.features.matched_inliers_keypoints[:25]
    kp2 = matches.frame2.features.matched_inliers_keypoints[:25]

    if img1.ndim == 2 or img1.shape[-1] == 1:
        img1 = cv2.cvtColor(matches.frame1.image, cv2.COLOR_GRAY2RGB)

    if img2.ndim == 2 or img2.shape[-1] == 1:
        img2 = cv2.cvtColor(matches.frame2.image, cv2.COLOR_GRAY2RGB)

    img = cv2.hconcat([img1, img2])

    offset = np.array([W, 0]).reshape(1, 2, 1)

    colors = (np.random.rand(len(kp1), 3) * 255).astype(int).tolist()
    colors = list(map(tuple, colors))

    img = draw_keypoints(img, kp1, colors=colors)
    img = draw_keypoints(img, kp2 + offset, colors=colors)
    img = draw_lines(
        img,
        kp1,
        kp2 + offset,
        colors=colors,
    )

    return img


def plot_keypoints(image: np.array, features: Features) -> np.array:
    """Display the keypoint count of the image.

    Args:
        image (np.array): The image to display the FPS on.
        features (float): The features object containing the keypoints.

    Returns:
        np.array: The image with the keypoint properties overlay.
    """
    if image.ndim == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = draw_keypoints(
        image,
        features.keypoints[features.state == 0],
        colors=(255, 0, 0),
    )
    image = draw_keypoints(
        image,
        features.keypoints[features.state == 1],
        colors=(0, 255, 255),
    )
    image = draw_keypoints(
        image,
        features.keypoints[features.state == 2],
        colors=(0, 255, 0),
    )

    # Draw tracks
    image = draw_keypoints(
        image,
        features.tracks[features.matched_candidate_inliers],
        colors=(100, 100, 100),
    )
    image = draw_lines(
        image,
        features.keypoints[features.candidate_mask],
        features.tracks[features.candidate_mask],
        colors=(0, 255, 0),
    )
    image = draw_lines(
        image,
        features.keypoints[
            features.matched_candidate_inliers & ~features.candidate_mask
        ],
        features.tracks[features.matched_candidate_inliers & ~features.candidate_mask],
        colors=(255, 255, 255),
    )

    return image
