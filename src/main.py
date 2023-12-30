import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque


from vo.primitives import Sequence, Features, Matches, State
from vo.pose_estimation import P3PPoseEstimator
from vo.landmarks import LandmarksTriangulator
from vo.features import KLTTracker, HarrisCornerDetector, Tracker
from vo.helpers import to_homogeneous_coordinates, to_cartesian_coordinates
from vo.visualization.overlays import (
    display_fps,
    display_keypoints_info,
    plot_keypoints,
    plot_matches,
)

from vo.visualization.point_cloud import PointCloudVisualizer
from vo.sensors import Camera


fig = plt.figure()
# fig.suptitle("Camera Trajectory", fontsize=16)

ax1 = fig.add_subplot(2, 4, (1, 2))
ax2 = fig.add_subplot(2, 4, (3, 8))
ax3 = fig.add_subplot(2, 4, 5)
ax4 = fig.add_subplot(2, 4, 6)
plt.ion()
plt.pause(1.0e-6)
plt.show()

# pc_visualizer = PointCloudVisualizer()

TRACKER_MODE = "harris"
DATA_SET = "kitti"
SHOW_N_POSES = 20


def plot_image(img):
    ax1.clear()
    ax1.imshow(img[::2, ::2])

    ax1.set_title("Current image", fontsize=8, fontweight="bold")
    # ax1.get_xaxis().set_visible(False)
    # ax1.get_yaxis().set_visible(False)

    fig.canvas.draw()
    fig.canvas.flush_events()


def plot_trajectory_with_landmarks(trajectory, landmarks):
    t_vec = trajectory[-SHOW_N_POSES:, :3, 3]

    # Extract x, y, z coordinates from the trajectory
    x = t_vec[:, 0]
    y = t_vec[:, 1]
    z = t_vec[:, 2]

    # Plot the camera trajectory
    ax2.clear()
    ax2.plot(x, z, marker="o", linewidth=1, markersize=1)
    ax2.scatter(landmarks[:, 0], landmarks[:, 2], marker="x", color="orange")

    # ax2.set_xlabel("X-axis")
    # ax2.set_ylabel("Z-axis")
    ax2.set_title(
        f"Trajectory of last {SHOW_N_POSES} frames and landmarks.",
        # f"(X, Y, Z) = ({x[-1]:.1f}, {y[-1]:.1f}, {z[-1]:.1f}), Landmarks: {len(landmarks)}",
        fontsize=8,
        fontweight="bold",
    )

    # fix the scaling of the axes
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_box_aspect(1.25)
    fig.canvas.draw()
    fig.canvas.flush_events()


def plot_trajectory(trajectory):
    t_vec = trajectory[:, :3, 3]

    # Extract x, y, z coordinates from the trajectory
    x = t_vec[:, 0]
    y = t_vec[:, 1]
    z = t_vec[:, 2]

    # Plot the camera trajectory
    ax4.clear()
    ax4.plot(x, z, marker="o", linewidth=1, markersize=1)

    # ax4.set_xlabel("X-axis")
    # ax4.set_ylabel("Z-axis")
    ax4.set_title("Full trajectory.", fontsize=8, fontweight="bold")

    # fix the scaling of the axes
    ax4.set_aspect("equal", adjustable="box")
    ax4.set_box_aspect(1.2)
    fig.canvas.draw()
    fig.canvas.flush_events()


def plot_nr_of_landmarks(nr_of_landmarks):
    x = np.arange(-SHOW_N_POSES, 0)
    y = nr_of_landmarks[-SHOW_N_POSES:]

    ax3.clear()
    ax3.plot(x, y, marker="o", linewidth=1, markersize=1)

    # ax3.set_xlabel("Frame")
    # ax3.set_ylabel("Nr of Landmarks")
    ax3.set_title(
        f"# tracked landmarks over last {SHOW_N_POSES} frames.",
        fontsize=8,
        fontweight="bold",
    )

    # fix the scaling of the axes
    # ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlim(-20, 0)
    ax3.set_ylim(0, 500)
    ax3.set_box_aspect(1.2)
    fig.canvas.draw()
    fig.canvas.flush_events()


def main():
    # Load sequence
    sequence = Sequence(DATA_SET)

    # Skipt to first curve
    # for _ in range(50):
    #     next(sequence)

    # 4x4 array, zero rotation and translation vector at origin
    current_pose = np.eye(4)
    current_pose[:3, 3] = np.array([0, 0, 0])
    current_pose[:3, :3] = np.eye(3)

    trajectory = []
    nr_of_landmarks = [0] * SHOW_N_POSES
    camera = sequence.get_camera()

    triangulator = LandmarksTriangulator(
        camera1=camera,
        camera2=camera,
        use_ransac=True,
        use_opencv=True,
        outlier_ratio=0.9,
        ransac_threshold=3.0,
        ransac_confidence=0.99,
    )
    pose_estimator = P3PPoseEstimator(
        use_opencv=False,
        intrinsic_matrix=camera.intrinsic_matrix,
        inlier_threshold=3,
        outlier_ratio=0.9,
        confidence=0.99,
        nonlinear_refinement=True,
    )

    # Perform bootstrapping
    init_frame = next(sequence)
    state = State(init_frame)

    next(sequence)  # skip frame 1
    new_frame = next(sequence)

    tracker = Tracker(init_frame, mode=TRACKER_MODE)
    matches = tracker.trackFeatures(state.curr_frame, new_frame)
    state.update_from_matches(matches)

    # Bootstrapping triangulation
    M, landmarks, inliers = triangulator.triangulate_matches(matches)
    # pc_visualizer.visualize_points(landmarks[inliers])

    outliers = np.zeros(shape=(matches.frame2.features.length,), dtype=bool)
    outliers[matches.frame2.features.match_inliers] = ~inliers

    state.update_with_local_pose(M)

    # Update landmarks (Note: order is important, always update pose before landmarks)
    inliers_mask = np.zeros_like(
        matches.frame2.features.matched_candidate_inliers
    ).astype(bool)
    inliers_mask[matches.frame2.features.matched_candidate_inliers] = inliers
    state.update_with_local_landmarks(landmarks[inliers], inliers_mask)
    state.reset_outliers(outliers)

    # Initialize plots
    trajectory.append(np.eye(4))
    trajectory.append(state.get_pose())
    nr_of_landmarks.append(
        len(state.curr_frame.features.triangulated_inliers_landmarks)
    )
    plot_trajectory_with_landmarks(
        np.array(trajectory),
        state.curr_frame.features.triangulated_inliers_landmarks,
    )
    plot_trajectory(np.array(trajectory))
    plot_nr_of_landmarks(np.array(nr_of_landmarks))

    # Queue to store last [maxlen] FPS
    fps_queue = deque([], maxlen=5)

    for new_frame in tqdm(sequence):
        # Variable for calculating FPS
        start_time = time.time()

        matches = tracker.trackFeatures(state.curr_frame, new_frame)

        (rmatrix, tvec), inliers = pose_estimator.estimate_pose(
            Features(
                keypoints=matches.frame2.features.triangulated_inliers_keypoints,
                landmarks=matches.frame2.features.triangulated_inliers_landmarks,  # landmarks=matches.frame1.features.landmarks,
            )  # 3D-2D correspondences
        )

        outliers = np.zeros(shape=(matches.frame2.features.length,), dtype=bool)
        outliers[matches.frame2.features.triangulate_inliers] = ~inliers

        # Update state
        state.update_from_matches(matches)
        state.update_with_world_pose(np.concatenate((rmatrix, tvec), axis=1))
        state.reset_outliers(outliers)  # before computing candidates reset all outliers
        state.compute_candidates()

        assert np.sum(state.curr_frame.features.candidate_mask) <= np.sum(
            state.curr_frame.features.matched_candidate_inliers
        )

        # Plot keypoints here because afterwards we triangulate all of them anyways
        img = plot_keypoints(new_frame.image, new_frame.features)
        img = display_keypoints_info(img, new_frame.features)
        # match_img = plot_matches(matches)

        # Triangulate new landmarks
        n_candidates = np.sum(state.curr_frame.features.candidate_mask)

        if n_candidates > 0:
            landmarks_world = triangulator.triangulate_candidates(
                state.curr_frame.features,
                current_pose=state.get_pose(),
            )
            state.update_with_world_landmarks(
                landmarks_world, matches.frame2.features.candidate_mask
            )
            # pc_visualizer.visualize_points(landmarks_world)

        # Update the trajectory and nr of landmarks arrays
        trajectory.append(state.get_pose())
        nr_of_landmarks.append(
            len(state.curr_frame.features.triangulated_inliers_landmarks)
        )
        # pc_visualizer.visualize_camera(
        #     camera=Camera(matches.frame2.intrinsics, R=rmatrix, t=tvec)
        # )

        # if len(trajectory) > SHOW_N_POSES:
        #     trajectory.pop(0)
        if len(nr_of_landmarks) > SHOW_N_POSES:
            nr_of_landmarks.pop(0)
        if len(trajectory) > 0:
            plot_trajectory_with_landmarks(
                np.array(trajectory),
                state.curr_frame.features.triangulated_inliers_landmarks,
            )
            plot_trajectory(np.array(trajectory))
            plot_nr_of_landmarks(np.array(nr_of_landmarks))

            # Display the resulting frame
            img, fps_queue = display_fps(
                image=img, start_time=start_time, fps_queue=fps_queue
            )
            plot_image(img)

        # cv2.imshow("Press esc to stop", img)
        # cv2.imshow("Press esc to stop", match_img)

        # k = cv2.waitKey(5) & 0xFF  # 30ms delay -> try lower value for more FPS :)
        # if k == 27:
        #     break


if __name__ == "__main__":
    main()
