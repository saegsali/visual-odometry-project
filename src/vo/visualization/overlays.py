import time
import cv2
import numpy as np


# Function to calculate and display FPS
def display_fps(image: np.array, start_time: float, frame_count: int) -> np.array:
    """Display the FPS on the image.

    Args:
        image (np.array): The image to display the FPS on.
        start_time (float): The time the processing started.
        frame_count (int): The number of frames processed.

    Returns:
        np.array: The image with the FPS overlay.
    """

    current_time = time.time()
    elapsed_time = current_time - start_time
    fps = frame_count / elapsed_time
    cv2.putText(
        image,
        f"FPS: {fps:.2f}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image
