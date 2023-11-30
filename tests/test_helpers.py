import numpy as np
import pytest

from vo.helpers import (
    normalize_points,
    to_cartesian_coordinates,
    to_homogeneous_coordinates,
    to_skew_symmetric_matrix,
)

# Set random seed generator
rng = np.random.default_rng(2023)


def test_to_homogeneous_coordinates():
    points = np.array([[[1], [2]], [[3], [4]]])
    expected_output = np.array([[[1], [2], [1]], [[3], [4], [1]]])

    assert np.array_equal(to_homogeneous_coordinates(points), expected_output)


def test_to_cartesian_coordinates():
    points = np.array([[[1], [2], [1]], [[3], [4], [2]]])
    expected_output = np.array([[[1], [2]], [[1.5], [2]]])

    assert np.array_equal(to_cartesian_coordinates(points), expected_output)


def test_to_cartesian_coordinates_zero_division():
    points = np.array([[[1], [2], [0]], [[3], [4], [0]]])

    with pytest.warns(RuntimeWarning):
        to_cartesian_coordinates(points)


def test_normalize_points2D():
    points = np.array([[[1], [2]], [[3], [4]], [[-1], [1]], [[0], [2]]])
    normalized_points, T = normalize_points(points)

    # Check that the mean of the normalized points is close to 0
    assert np.allclose(np.mean(normalized_points, axis=0), 0)

    # Check that the average distance of the normalized points is close to sqrt(2)
    assert np.allclose(
        np.sqrt(np.mean(np.sum(normalized_points**2, axis=-2))), np.sqrt(2)
    )

    # Check that applying the transformation matrix to the points gives the normalized points
    assert np.allclose(
        normalized_points,
        to_cartesian_coordinates((T @ to_homogeneous_coordinates(points).T).T),
    )


def test_normalize_points3D():
    points = np.array([[[1], [2], [3]], [[3], [4], [5]], [[0], [0], [0]]])
    print(points.shape)
    normalized_points, T = normalize_points(points)

    # Check that the mean of the normalized points is close to 0
    assert np.allclose(np.mean(normalized_points, axis=0), 0)

    # Check that the average distance of the normalized points is close to sqrt(3)
    assert np.allclose(
        np.sqrt(np.mean(np.sum(normalized_points**2, axis=-2))), np.sqrt(3)
    )

    # Check that applying the transformation matrix to the points gives the normalized points
    assert np.allclose(
        normalized_points,
        to_cartesian_coordinates((T @ to_homogeneous_coordinates(points).T).T),
    )


def test_normalize_points_complex():
    points = rng.normal(-3, 10, size=(1000, 2, 1))
    normalized_points, T = normalize_points(points)

    # Check that the mean of the normalized points is close to 0
    assert np.allclose(np.mean(normalized_points, axis=0), 0)

    # Check that the average distance of the normalized points is close to sqrt(2)
    assert np.allclose(
        np.sqrt(np.mean(np.sum(normalized_points**2, axis=-2))), np.sqrt(2)
    )

    # Check that applying the transformation matrix to the points gives the normalized points
    assert np.allclose(
        normalized_points,
        to_cartesian_coordinates((T @ to_homogeneous_coordinates(points).T).T),
    )


def test_to_skew_symmetric_matrix():
    vector = np.array([[1, 2, 3]]).T
    expected_output = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])

    assert np.array_equal(to_skew_symmetric_matrix(vector), expected_output)

    vector_batch = np.array([[[1, 2, 3]], [[4, 5, 6]]]).reshape(2, 3, 1)
    expected_output_batch = np.array(
        [[[0, -3, 2], [3, 0, -1], [-2, 1, 0]], [[0, -6, 5], [6, 0, -4], [-5, 4, 0]]]
    )

    assert np.array_equal(to_skew_symmetric_matrix(vector_batch), expected_output_batch)


if __name__ == "__main__":
    pytest.main(["-v"])
