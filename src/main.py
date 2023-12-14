import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


from vo.primitives import Sequence, Features, Matches, State
from vo.pose_estimation import P3PPoseEstimator
from vo.landmarks import LandmarksTriangulator
from vo.features import KLTTracker
from vo.helpers import to_homogeneous_coordinates, to_cartesian_coordinates
from vo.visualization.overlays import display_fps


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.ion()
plt.show()

USE_KLT = True


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


def get_sift_matches(frame1, frame2, force_recompute_frame1=False):
    # Compute matches using SIFT (TODO: replace later with our FeatureMatcher)
    sift = cv2.SIFT_create()
    if force_recompute_frame1 or frame1.features is None:
        keypoints1_raw, descriptors1 = sift.detectAndCompute(frame1.image, None)
        keypoints1 = np.array([kp.pt for kp in keypoints1_raw]).reshape(-1, 2, 1)
        descriptors1 = np.array(descriptors1)
        landmarks1 = None
    else:
        keypoints1 = frame1.features.keypoints
        descriptors1 = frame1.features.descriptors
        landmarks1 = frame1.features.landmarks

    keypoints2_raw, descriptors2 = sift.detectAndCompute(frame2.image, None)
    keypoints2 = np.array([kp.pt for kp in keypoints2_raw]).reshape(-1, 2, 1)
    descriptors2 = np.array(descriptors2)

    frame1.features = Features(keypoints1, descriptors1, landmarks=landmarks1)
    frame2.features = Features(keypoints2, descriptors2)

    # Match keypoints
    index_params = dict(algorithm=0, trees=20)
    search_params = dict(checks=150)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # # Need to draw only good matches, so create a mask
    # good_matches = [[0, 0] for i in range(len(matches))]

    # # Good matches
    # for i, (m, n) in enumerate(matches):
    #     if m.distance < 0.5 * n.distance:
    #         good_matches[i] = [1, 0]

    # # plot matches
    # img = cv2.drawMatchesKnn(
    #     frame1.image,
    #     keypoints1_raw,
    #     frame2.image,
    #     keypoints2_raw,
    #     matches,
    #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    # )
    # cv2.imshow("Press esc to stop", img)
    # k = cv2.waitKey(30) & 0xFF  # 30ms delay -> try lower value for more FPS :)
    # if k == 27:
    #     break

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m.queryIdx, m.trainIdx])
    good = np.array(good)

    # Visualize fundamental matrix
    matches = Matches(frame1, frame2, matches=good)
    return matches


def main():
    # Load sequence
    sequence = Sequence("kitti")
    frame1 = next(sequence)

    # klt_tracker = KLTTracker(frame1)

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

    if USE_KLT:
        klt = KLTTracker(init_frame)
        matches = klt.track_features(new_frame)
    else:
        matches = get_sift_matches(init_frame, new_frame)

    M, landmarks, inliers = triangulator.triangulate_matches(matches)

    # Update state
    state.update_from_matches(matches)
    state.update_with_local_landmarks(landmarks, inliers=inliers)
    state.update_with_local_pose(M)

    # Variables for calculating FPS
    start_time = time.time()

    for new_frame in tqdm(sequence):
        if USE_KLT:
            matches = klt.track_features(new_frame)
        else:
            matches = get_sift_matches(
                state.get_frame(), new_frame, force_recompute_frame1=True
            )  # TODO: keep keypoints of current frame and do not detect new ones to simulate "tracking" by setting force_recompute_frame1=False

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
        img = display_fps(
            image=new_frame.image,
            start_time=start_time,
            frame_count=new_frame.get_frame_id(),
        )
        cv2.imshow("Press esc to stop", img)

        k = cv2.waitKey(5) & 0xFF  # 30ms delay -> try lower value for more FPS :)
        if k == 27:
            break


if __name__ == "__main__":
    main()
