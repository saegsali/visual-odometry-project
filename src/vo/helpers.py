import numpy as np


def to_homogeneous_coordinates(points: np.ndarray) -> np.ndarray:
    """Convert points to homogeneous coordinates.

    Args:
        points (np.ndarray): array of points, shape = (N, D).

    Returns:
        np.ndarray: array of points in homogeneous coordinates, shape = (N, D+1).
    """
    return np.concatenate((points, np.ones(shape=(points.shape[0], 1))), axis=-1)


def to_cartesian_coordinates(points: np.ndarray) -> np.ndarray:
    """Convert points from homogeneous back to cartesian coordinates.

    Args:
        points (np.ndarray): array of points, shape = (N, D+1).

    Returns:
        np.ndarray: array of points in cartesian coordinates, shape = (N, D).
    """
    return points[:, :-1] / points[:, -1:]


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Normalizes points in cartesian coordintates and returns normalization matrix.

    Args:
        points (np.ndarray): Points to normalize, shape = (N, D)

    Returns:
        tuple[np.ndarray, np.ndarray]: Returns transformed points (N, D)
                                       and transformation matrix of shape (D+1, D+1) using homogenous coordinates.
    """
    N = points.shape[0]
    D = points.shape[1]

    # Compute centroids and sigma
    mu = np.mean(points, axis=0, keepdims=True)
    sigma = np.sqrt(np.mean(np.sum((points - mu) ** 2, axis=-1)))
    s = np.sqrt(D) / sigma

    # Compute transformation matrix
    T = np.diag([s] * D + [1])
    T[:-1, -1:] = -s * mu.reshape(D, 1)

    pts_tilde = (T @ to_homogeneous_coordinates(points).T).T
    pts_tilde = to_cartesian_coordinates(pts_tilde)
    return pts_tilde, T
