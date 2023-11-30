import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits
import pytransform3d.camera as pc
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as pu

from vo.sensors import Camera


class PointCloudVisualizer:
    def __init__(self) -> None:
        plt.ion()
        self.fig = plt.figure(figsize=(5, 5))
        self.ax = pu.make_3d_axis(ax_s=1, unit="m")
        # self.ax.autoscale(enable=True, axis="both", tight=True)

        self.x = np.zeros((1, 1))
        self.y = np.zeros((1, 1))
        self.z = np.zeros((1, 1))

        plt.show()

    def _draw(self, percentiles=[5, 95]) -> None:
        self.ax.view_init(elev=-70, azim=-80, roll=0)

        # Rescale axes
        self.ax.set_xlim(np.percentile(self.x, percentiles))
        self.ax.set_ylim(np.percentile(self.y, percentiles))
        self.ax.set_zlim(np.percentile(self.z, percentiles))

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
            alpha=0.1,
        )
        self.x = np.append(self.x, points[:, 0].flatten())
        self.y = np.append(self.y, points[:, 1].flatten())
        self.z = np.append(self.z, points[:, 2].flatten())
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
