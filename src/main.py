import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


from vo.primitives import Sequence, Features, Matches
from vo.pose_estimation import P3PPoseEstimator
from vo.landmarks import LandmarksTriangulator
from vo.features import KLTTracker


def plot_trajectory(trajectory):
    t_vec = trajectory[:, :3, 3]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Extract x, y, z coordinates from the trajectory
    x = t_vec[:, 0]
    y = t_vec[:, 1]
    z = t_vec[:, 2]

    # Plot the camera trajectory
    ax.plot(x, y, z, marker="o")

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("Camera Trajectory")

    # fix the scaling of the axes
    ax.set_aspect("equal", adjustable="box")

    plt.show()


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

    for frame2 in tqdm(sequence):
        if frame2.get_frame_id() < 80:
            frame1 = frame2
            continue

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

        # Compute matches using SIFT (TODO: replace later with our FeatureMatcher)
        sift = cv2.SIFT_create()
        keypoints1_raw, descriptors1 = sift.detectAndCompute(frame1.image, None)
        keypoints2_raw, descriptors2 = sift.detectAndCompute(frame2.image, None)

        keypoints1 = np.array([kp.pt for kp in keypoints1_raw]).reshape(-1, 2, 1)
        keypoints2 = np.array([kp.pt for kp in keypoints2_raw]).reshape(-1, 2, 1)
        descriptors1 = np.array(descriptors1)
        descriptors2 = np.array(descriptors2)

        frame1.features = Features(keypoints1, descriptors1)
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

        # # klt = KLTTracker(frame1)
        # # Calculate optical flow
        # matches = klt_tracker.track_features(frame2)

        # # Draw the tracks on the image
        # image, mask = klt_tracker.draw_tracks()
        # # Display the resulting frame
        # cv2.imshow("Press esc to stop", image)

        # k = cv2.waitKey(30) & 0xFF  # 30ms delay -> try lower value for more FPS :)
        # if k == 27:
        #     break

        if frame2.get_frame_id() < 2:
            continue

        M, landmarks, inliers = triangulator.triangulate_matches(matches)

        # landmarks to homogeneous coordinates form (N, 3, 1) to (N, 4, 1)
        landmarks = np.concatenate(
            (landmarks, np.ones((landmarks.shape[0], 1, 1))), axis=1
        )

        # project landmarks back to frame1
        landmarks = np.linalg.inv(current_pose) @ landmarks

        # back to non-homogeneous coordinates
        landmarks = landmarks[:, :3, :] / np.expand_dims(landmarks[:, 3, :], axis=2)

        matches.frame2.features.landmarks = landmarks
        matches.frame2.features.apply_inliers(inliers)
        model, inliers = pose_estimator.estimate_pose(matches.frame2.features)
        rmatrix, tvec = model
        # compare rmatrix and tvec with ground truth
        # print("3D-2D Rotation matrix: ", rmatrix)
        # print("2D-2D rotation matrix: ", M[:3, :3])
        # print("3D-2D Translation vector: ", tvec[:, 0])
        # print("2D-2D translation vector: ", M[:3, 3])

        # Update the current pose with rmatrix, tvec
        current_pose = np.eye(4)
        # current_pose[:3, 3] = M[:3, 3]
        # current_pose[:3, :3] = M[:3, :3]
        current_pose[:3, 3] = tvec[:, 0]
        current_pose[:3, :3] = rmatrix

        # Update the trajectory array
        trajectory.append(current_pose)

        if len(trajectory) > 0:
            # Plot the trajectory every 5 frames
            if frame2.get_frame_id() % 5 == 0:
                plot_trajectory(np.array(trajectory))

        # Display the resulting frame
        cv2.imshow("Press esc to stop", frame2.image)

        k = cv2.waitKey(30) & 0xFF  # 30ms delay -> try lower value for more FPS :)
        if k == 27:
            break

        frame1 = frame2


if __name__ == "__main__":
    main()
