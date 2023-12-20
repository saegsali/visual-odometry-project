import time
import cv2
import numpy as np


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
