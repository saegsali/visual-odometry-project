import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.camera as pc
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as pu

from vo.sensors import Camera


class PointCloudVisualizer:
    def __init__(self) -> None:
        plt.ion()
        self.fig = plt.figure(figsize=(5, 5))
        self.ax = pu.make_3d_axis(ax_s=1, unit="m")
        plt.show()

    def _draw(self) -> None:
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.view_init(elev=-70, azim=-80, roll=0)
        plt.draw()

    def visualize_points(self, points: np.ndarray, color="b") -> None:
        """Add points to axes.

        Args:
            points (np.ndarray): List of 3D points to visualize.
        """
        self.ax.scatter(
            points[:, 0].flatten(),
            points[:, 1].flatten(),
            points[:, 2].flatten(),
            color=color,
        )
        self._draw()

    def visualize_camera(self, camera: Camera) -> None:
        """Add camera to axes.

        Args:
            camera (Camera): Camera to visualize.
        """
        pt.plot_transform(self.ax, np.linalg.inv(camera.c_T_w), s=0.1)
        pc.plot_camera(
            self.ax,
            cam2world=np.linalg.inv(camera.c_T_w),
            M=camera.intrinsic_matrix,
            virtual_image_distance=0.1,
            sensor_size=(480, 320),
        )
        self._draw()
