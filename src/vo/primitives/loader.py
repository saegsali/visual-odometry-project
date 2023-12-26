import os
import glob
import cv2
import configparser
import numpy as np

from vo.primitives import Frame
from vo.sensors import Camera


class Sequence:
    """A class to load images from a directory.

    Args:
        dataset (str): The dataset to load images from. Currently supports `kitti`, `malaga` and `parking`.
        path (str): The path to the data folder relative to the project root.
        camera (int): The camera to use if supported by the dataset. Left camera is 0, right camera is 1.
        increment (int): The number of frames to skip between each frame. Default is 1.
    """

    project_name = "visual-odometry-project"

    def __init__(
        self,
        dataset: str,
        path: str = "./data",
        camera: int = 0,
        increment: int = 1,
        rectified: bool = False,
        use_lowres: bool = False,
    ):
        self.dataset = dataset
        self._rel_data_path = path
        self.data_dir = self.get_data_dir()
        self.camera = camera
        self.intrinsics = None
        self.idx = 0
        self.increment = increment
        self._rectified = rectified
        self._use_lowres = use_lowres
        self.images = self._load()
        self.sensor = Camera(intrinsic_matrix=self.intrinsics)

    def get_data_dir(self):
        path = os.path.dirname(os.path.realpath(__file__))
        # clip everything after viual_odometry
        root_dir = path.split(self.project_name)
        directory = root_dir[0]
        if (
            len(root_dir) > 2
        ):  # if the project name is in the path twice. Happens in GitHub actions
            directory = os.path.join(directory, self.project_name)
        return os.path.join(directory, self.project_name, self._rel_data_path)

    def _load(self):
        """Load images from a directory.

        Returns:
            list: A list of Frame objects.
        """
        match self.dataset:
            case "kitti":
                images = self._load_kitti()
            case "malaga":
                images = self._load_malaga()
            case "parking":
                images = self._load_parking()
            case _:
                raise Exception("Invalid dataset")
        return images

    def _load_kitti(self):
        """Load images from the KITTI dataset.

        Returns:
            list: A list of paths to the images.
        """
        print("path: ", self.data_dir)
        # get image paths
        data_path = os.path.join(self.data_dir, "kitti", "05", f"image_{self.camera}")
        image_paths = sorted(glob.glob(data_path + "/*.png"))
        print("Loading {} images from {}".format(len(image_paths), data_path))

        # load intrinsics
        intrinsics_file = os.path.join(self.data_dir, "kitti", "05", "calib.txt")
        with open(intrinsics_file, "r") as file:
            calib_lines = file.readlines()
            self.intrinsics = calib_matrix_line = calib_lines[2 * self.camera + 1]
            calib_matrix_values = calib_matrix_line.split(" ")[1:]
            calib_matrix_values[-1] = calib_matrix_values[-1].split("\n")[0]
            self.intrinsics = np.array(
                [np.float32(value) for value in calib_matrix_values]
            ).reshape(3, 4)[:, :3]

        return image_paths

    def _load_malaga(self):
        """Load images from the Malaga dataset.

        Returns:
            list: A list of paths to the images.
        """
        # get image paths
        data_path = os.path.join(
            self.data_dir, "malaga-urban-dataset-extract-07", "Images"
        )
        if self.camera == 0:  # use left camera
            image_paths = sorted(glob.glob(data_path + "/*left.jpg"))
        else:  # use right camera
            image_paths = sorted(glob.glob(data_path + "/*right.jpg"))

        print("Loading {} images from {}".format(len(image_paths), data_path))

        # load intrinsics
        if not self._rectified:
            config_file = "camera_params_raw_1024x768.txt"
            match self.camera:
                case 0:
                    camera = "CAMERA_PARAMS_LEFT"
                case 1:
                    camera = "CAMERA_PARAMS_RIGHT"
        else:  # rectified
            if self._use_lowres:
                config_file = "camera_params_rectified_a=0_800x600.txt"
            else:
                config_file = "camera_params_rectified_a=0_1024x768.txt"
            match self.camera:
                case 0:
                    camera = "CAMERA_LEFT"
                case 1:
                    camera = "CAMERA_RIGHT"

        intrinsics_file = os.path.join(
            self.data_dir,
            "malaga-urban-dataset-extract-07",
            config_file,
        )
        config = configparser.ConfigParser()
        config.read(intrinsics_file)

        intrinsics = config[camera]
        self.intrinsics = np.array(
            [
                [
                    np.float32(intrinsics["fx"].split("//")[0]),
                    0,
                    np.float32(intrinsics["cx"].split("//")[0]),
                ],
                [
                    0,
                    np.float32(intrinsics["fy"].split("//")[0]),
                    np.float32(intrinsics["cy"].split("//")[0]),
                ],
                [0, 0, 1],
            ]
        )

        return image_paths

    def _load_parking(self):
        """Load images from the Parking dataset.

        Returns:
            list: A list of paths to the images.
        """
        # get image paths
        data_path = os.path.join(self.data_dir, "parking", "images")
        image_paths = sorted(glob.glob(data_path + "/*.png"))

        print("Loading {} images from {}".format(len(image_paths), data_path))

        # load intrinsics
        intrinsics_file = os.path.join(self.data_dir, "parking", "K.txt")
        with open(intrinsics_file, "r") as file:
            file_content = file.read()
            # Remove whitespaces and newlines
            file_content = file_content.replace(" ", "")
            file_content = file_content.replace("\n", "")
            self.intrinsics = (
                np.asarray(file_content.split(",")).astype(np.float32).reshape(3, 3)
            )
        return image_paths

    def get_frame(self, idx: int):
        """Return a frame at a given index.

        Args:
            idx (int): The index of the frame to return.

        Returns:
            Frame: The frame at the given index.
        """
        image = cv2.imread(self.images[idx])
        frame = Frame(image)
        frame.frame_id = idx
        frame.intrinsics = self.intrinsics
        frame.sensor = Camera(self.intrinsics)
        return frame

    def get_intrinsics(self) -> np.ndarray:
        """Return the camera intrinsics matrix.

        Returns:
            np.ndarray: The intrinsics matrix. (3x3)
        """
        return self.intrinsics

    def get_camera(self) -> Camera:
        """Return the camera object.

        Returns:
            Camera: The camera object.
        """
        return self.sensor

    def __len__(self) -> int:
        return len(self.images)

    def __next__(self) -> Frame:
        """Return the next frame in the dataset.

        Returns:
            Frame: The next frame in the dataset.
        """
        if self.idx >= len(self.images):
            raise StopIteration
        frame = self.get_frame(self.idx)
        self.idx += self.increment
        return frame

    def __iter__(self):
        """Return an iterator over the dataset.

        Returns:
            DataLoader: An iterator over the dataset.
        """
        return self

    def __repr__(self) -> str:
        return "Sequence(dataset={}, path={}, camera={}, increment={})".format(
            self.dataset, self._rel_data_path, self.camera, self.increment
        )


if __name__ == "__main__":
    # A simple example of how to use the Sequence class:
    video = Sequence(
        "malaga", increment=2, rectified=True, camera=1, use_lowres=True
    )  # Choose dataset and increment
    frame = video.get_frame(0)  # Get a frame at a given index

    print("Camera intrinsics:\n", frame.get_intrinsics())  # Get the intrinsics matrix
    # print(video.get_intrinsics())  # ...this also works (for now)

    for i in range(10):
        last_frame = frame
        frame = next(video)  # Get the next frame (with increment)
        print(frame)
        frame.show(last_frame)  # Show the frame with the previous frame
        frame.show()  # Show the (current) frame
