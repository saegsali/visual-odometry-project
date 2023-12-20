from vo.primitives import Features, Frame, Matches
from vo.features import KLTTracker, HarrisCornerDetector


class Tracker:
    """A class to choose the featrue tracker.

    Args:
        frame (Frame): The init frame to track features in.
        mode: Name of the desired tracker.

    Methods:
        initTracker: Initialise the chosen tracker.
        trackFeatures: Call the chose tracker and return matches object containing the matched features and descriptors.
    """

    def __init__(
        self,
        frame,
        mode = "klt"

        # frame2: Frame = None,
        # patch_size: int = 9,
        # kappa: float = 0.08,
        # num_keypoints: int = 200,
        # nonmaximum_supression_radius: int = 5,
        # descriptor_radius: int = 9,
        # match_lambda: float = 5.0,
    ):
        """Set parameters and Initialize selected tracker."""
        self._init_frame = frame
        self._mode = mode
        self._tracker = None
        self.initTracker(frame)
        
        # self._frame2 = frame2
        # self._patch_size = patch_size
        # self._kappa = kappa
        # self._num_keypoints = num_keypoints
        # self._nonmaximum_supression_radius = nonmaximum_supression_radius
        # self._descriptor_radius = descriptor_radius
        # self._match_lambda = match_lambda

    def initTracker(self, frame: Frame) -> Matches:
        match self._mode:
            case "klt":
                self._tracker =  KLTTracker(frame)
            case "harris":
                self._tracker =  HarrisCornerDetector(frame)
            case other:
                raise Exception("Tracker Name not valid")

    def trackFeatures(self, frame: Frame) -> Matches:
        match self._mode:
            case "klt":
                return self._tracker.track_features(frame)
            case "harris":
                return self._tracker.featureMatcher(frame)
            case other:
                raise Exception("Tracker Name not valid")