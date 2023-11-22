import os
import pytest
import numpy as np
from vo.primitives import Frame
from vo.primitives import Sequence


@pytest.fixture
def kitti_sequence():
    return Sequence("kitti", path="./data", camera=0, increment=1)


@pytest.fixture
def malaga_sequence():
    return Sequence("malaga", path="./data", camera=0, increment=1)


@pytest.fixture
def parking_sequence():
    return Sequence("parking", path="./data", camera=0, increment=1)


def test_kitti_sequence_length(kitti_sequence):
    assert len(kitti_sequence) > 0


def test_malaga_sequence_length(malaga_sequence):
    assert len(malaga_sequence) > 0


def test_parking_sequence_length(parking_sequence):
    assert len(parking_sequence) > 0


def test_get_frame(kitti_sequence):
    frame = kitti_sequence.get_frame(0)
    assert isinstance(frame, Frame)


def test_get_intrinsics(kitti_sequence):
    intrinsics = kitti_sequence.get_intrinsics()
    assert isinstance(intrinsics, np.ndarray)
    assert intrinsics.shape == (3, 3)


def test_next_frame(kitti_sequence):
    frame1 = next(kitti_sequence)
    frame2 = next(kitti_sequence)
    assert frame1.frame_id < frame2.frame_id


if __name__ == "__main__":
    import pytest

    pytest.main(["-v", __file__])
