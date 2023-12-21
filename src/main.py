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
from vo.visualization.overlays import display_fps


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.ion()
plt.pause(1.0e-6)
plt.show()

TRACKER_MODE = "harris"


def plot_trajectory(trajectory):
    t_vec = trajectory[:, :3, 3]

    # Extract x, y, z coordinates from the trajectory
    x = t_vec[:, 0]
    # y = t_vec[:, 1]
    z = t_vec[:, 2]

    # Plot the camera trajectory
    ax.clear()
    ax.plot(x, z, marker="o")

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Z-axis")
    ax.set_title("Camera Trajectory")

    # fix the scaling of the axes
    ax.set_aspect("equal", adjustable="box")
    fig.canvas.draw()
    fig.canvas.flush_events()


def main():
    # Load sequence
    sequence = Sequence("kitti")
    frame1 = next(sequence)

    # 4x4 array, zero rotation and translation vector at origin
    current_pose = np.eye(4)
    current_pose[:3, 3] = np.array([0, 0, 0])
    current_pose[:3, :3] = np.eye(3)

    trajectory = []
    camera = sequence.get_camera()

    triangulator = LandmarksTriangulator(
        camera1=camera, camera2=camera, use_opencv=True
    )
    pose_estimator = P3PPoseEstimator(
        intrinsic_matrix=camera.intrinsic_matrix,
        inlier_threshold=1,
        outlier_ratio=0.9,
        confidence=0.99,
        max_iterations=1000,
    )

    # Perform bootstrapping
    init_frame = next(sequence)
    state = State(init_frame)

    next(sequence)  # skip frame 1
    new_frame = next(sequence)

    tracker = Tracker(init_frame, mode=TRACKER_MODE)
    matches = tracker.trackFeatures(new_frame)
    # if USE_KLT:
    #     klt = KLTTracker(init_frame)
    #     matches = klt.track_features(new_frame)
    # elif USE_HARRIS:
    #     harris = HarrisCornerDetector(init_frame)
    #     matches = harris.featureMatcher(new_frame)
    #     last_frame = new_frame
    # else:
    #     matches = get_sift_matches(init_frame, new_frame)

    M, landmarks, inliers = triangulator.triangulate_matches(matches)

    # Update state
    state.update_from_matches(matches)
    state.update_with_local_landmarks(landmarks, inliers=inliers)
    state.update_with_local_pose(M)

    # Queue to store last [maxlen] FPS
    fps_queue = deque([], maxlen=5)

    for new_frame in tqdm(sequence):
        # Variable for calculating FPS
        start_time = time.time()

        matches = tracker.trackFeatures(new_frame)
        # if USE_KLT:
        #     matches = klt.track_features(new_frame)
        # elif USE_HARRIS:
        #     matches = harris.featureMatcher(new_frame)
        #     last_frame = new_frame
        # else:
        #     matches = get_sift_matches(
        #         state.get_frame(), new_frame, force_recompute_frame1=True
        #     )  # TODO: keep keypoints of current frame and do not detect new ones to simulate "tracking" by setting force_recompute_frame1=False

        # # FIXME: This step is not allowed in continuous operation, but triangulation below should be used
        M, landmarks, inliers = triangulator.triangulate_matches(matches)
        matches.apply_inliers(inliers)
        landmarks = landmarks[inliers]

        # Ideally we would reuse landmarks/keypoints from previous frame and match only existing ones with new frame
        # and can then use p3p without triangulation first
        (rmatrix, tvec), inliers = pose_estimator.estimate_pose(
            Features(
                keypoints=matches.frame2.features.keypoints,
                landmarks=to_cartesian_coordinates(
                    state.get_pose() @ to_homogeneous_coordinates(landmarks)
                ),  # landmarks=matches.frame1.features.landmarks,
            )  # 3D-2D correspondences
        )

        # # Triangulate new landmarks (in future from candidate keypoints, here we detect and recompute all keypoints)
        # matches_all = get_sift_matches(
        #     state.get_frame(), new_frame, force_recompute_frame1=True
        # )
        # landmarks_prev_frame = triangulator.triangulate_matches_with_relative_pose(
        #     matches_all, T=np.linalg.inv(state.curr_pose) @ state.prev_pose
        # )

        # Update state
        state.update_from_matches(matches)
        state.update_with_world_pose(np.concatenate((rmatrix, tvec), axis=1))
        state.update_with_local_landmarks(landmarks, inliers=inliers)
        # state.update_with_world_landmarks(
        #     to_cartesian_coordinates(
        #         state.prev_pose @ to_homogeneous_coordinates(landmarks_prev_frame)
        #     )
        # )

        # Update the trajectory array
        trajectory.append(state.get_pose())

        if len(trajectory) > 0:
            # Plot the trajectory every 5 frames
            # if frame2.get_frame_id() % 5 == 0:
            plot_trajectory(np.array(trajectory))

        # Display the resulting frame
        img, fps_queue = display_fps(
            image=new_frame.image,
            start_time=start_time,
            fps_queue = fps_queue
        )
        cv2.imshow("Press esc to stop", img)

        k = cv2.waitKey(5) & 0xFF  # 30ms delay -> try lower value for more FPS :)
        if k == 27:
            break


if __name__ == "__main__":
    main()
