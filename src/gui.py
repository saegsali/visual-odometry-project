from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    # QPushButton,
    # QFileDialog,
    # QHBoxLayout,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QRunnable, pyqtSlot, QObject, pyqtSignal, QThreadPool

# import cv2
import time
import sys
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import run_vo
from vo.primitives import Frame


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    """

    frame_signal = pyqtSignal(Frame)
    trajectory_signal = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self):
        super(Worker, self).__init__()

        # Initialise signals and slots
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        run_vo.run(self.signals)

    # terminate the thread
    def stop(self):
        self.signals.frame_signal.disconnect()
        self.signals.trajectory_signal.disconnect()


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self._init_ui()

        self.frame = None
        self.trajectory = []

        self.threadpool = QThreadPool()
        self.worker = Worker()
        self.worker.signals.frame_signal.connect(self.update_frame)
        self.worker.signals.trajectory_signal.connect(self.update_trajectory)
        self.threadpool.start(self.worker)

        self.show()

    def closeEvent(self, event):
        # Stop the worker thread before closing the GUI
        self.threadpool.clear()
        self.worker.stop()

        # Call the parent class method to handle the close event
        super(MainWindow, self).closeEvent(event)

    # update frame
    def update_frame(self, frame):
        self.frame = frame
        self._update_image()
        # self.frame = next(self.video)

    def update_trajectory(self, pose):
        self.trajectory.append(pose)
        self._update_trajectory_plot()

    def select_dataset(self, dataset):
        pass

    def _init_ui(self):
        # Set window title and size
        self.setWindowTitle("Visual Odometry GUI")
        self.setGeometry(100, 100, 1200, 800)

        # Create a central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create a vertical layout
        layout = QVBoxLayout()

        # button_layout = QHBoxLayout()

        # self.kitti_button = QPushButton("Kitti", self)
        # self.kitti_button.clicked.connect(self.select_dataset)
        # button_layout.addWidget(self.kitti_button)

        # self.parking_button = QPushButton("Parking", self)
        # self.parking_button.clicked.connect(self.select_dataset)
        # button_layout.addWidget(self.parking_button)

        # self.malaga_button = QPushButton("Malaga", self)
        # self.malaga_button.clicked.connect(self.select_dataset)
        # button_layout.addWidget(self.malaga_button)

        # # Add the button layout to the main layout
        # layout.addLayout(button_layout)

        # Labels for displaying images
        self.image_label = QLabel(self)

        layout.addWidget(self.image_label)

        # Matplotlib plot for displaying trajectory
        self.trajectory_figure = Figure()
        self.trajectory_canvas = FigureCanvas(self.trajectory_figure)
        layout.addWidget(self.trajectory_canvas)

        central_widget.setLayout(layout)

    def _update_image(self):
        if self.frame is None:
            return

        image_data = self.frame.image
        # keypoints = self.frame.features.keypoints
        # # draw keypoints
        # for kp in keypoints:
        #     cv2.circle(
        #         image_data,
        #         (int(kp[0]), int(kp[1])),
        #         5,
        #         (0, 255, 0),
        #         -1,
        #     )

        # Convert image array to QImage
        height, width, _ = image_data.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            image_data.data, width, height, bytes_per_line, QImage.Format_RGB888
        )

        # Display the image
        pixmap = QPixmap(q_image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)

    def _update_trajectory_plot(self):
        trajectory_data = self.trajectory

        # Return if no trajectory data is available
        if len(trajectory_data) == 0:
            return

        trajectory_data = np.array(trajectory_data)
        t_vec = trajectory_data[:, :3, 3]

        # Clear existing plot
        self.trajectory_figure.clear()

        # Plot new trajectory
        ax = self.trajectory_figure.add_subplot(1, 1, 1)
        ax.plot(
            t_vec[:, 0],
            # t_vec[:, 1],
            t_vec[:, 2],
            color="blue",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        # ax.set_zlabel("Z")

        ax.set_aspect("equal", adjustable="box")

        # Draw the plot
        self.trajectory_canvas.draw()


app = QApplication([])
window = MainWindow()
sys.exit(app.exec_())
