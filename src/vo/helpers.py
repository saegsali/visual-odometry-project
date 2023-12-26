import numpy as np
from scipy.linalg import expm, logm


def to_homogeneous_coordinates(points: np.ndarray) -> np.ndarray:
    """Convert points to homogeneous coordinates.

    Args:
        points (np.ndarray): array of points, shape = (N, D,1 ).

    Returns:
        np.ndarray: array of points in homogeneous coordinates, shape = (N, D+1, 1).
    """
    assert points.ndim == 3, "Points must have three dimensions"
    return np.concatenate((points, np.ones(shape=(points.shape[0], 1, 1))), axis=-2)


def to_cartesian_coordinates(points: np.ndarray) -> np.ndarray:
    """Convert points from homogeneous back to cartesian coordinates.

    Args:
        points (np.ndarray): array of points, shape = (N, D+1, 1).

    Returns:
        np.ndarray: array of points in cartesian coordinates, shape = (N, D, 1).
    """
    assert points.ndim == 3, "Points must have three dimensions"
    return points[:, :-1] / points[:, -1:]


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Normalizes points in cartesian coordintates and returns normalization matrix.

    Args:
        points (np.ndarray): Points to normalize, shape = (N, D, 1)

    Returns:
        tuple[np.ndarray, np.ndarray]: Returns transformed points (N, D, 1)
                                       and transformation matrix of shape (D+1, D+1) using homogenous coordinates.
    """
    D = points.shape[1]

    # Compute centroids and sigma
    mu = np.mean(points, axis=0, keepdims=True)
    sigma = np.sqrt(np.mean(np.sum((points - mu) ** 2, axis=-2)))
    s = np.sqrt(D) / sigma

    # Compute transformation matrix
    T = np.diag([s] * D + [1])
    T[:-1, -1:] = -s * mu.reshape(D, 1)

    pts_tilde = T @ to_homogeneous_coordinates(points)
    pts_tilde = to_cartesian_coordinates(pts_tilde)
    return pts_tilde, T


def to_skew_symmetric_matrix(v: np.ndarray) -> np.ndarray:
    """Converts a 3D vector to a skew-symmetric matrix,
    which can be used to compute the cross product with another vector.

    Args:
        v (np.ndarray): 3D vector or array of 3D vectors. Shape = (3, 1) or (N, 3, 1).

    Returns:
        np.ndarray: 3x3 skew-symmetric matrix.
    """
    assert (v.ndim == 2 and v.shape == (3, 1)) or (
        v.ndim == 3 and v.shape[1:] == (3, 1)
    ), "Vector must be a single 3D vector or an array of 3D vectors"

    N = v.shape[0] if v.ndim == 3 else 1
    out_array = np.zeros((N, 3, 3))
    out_array[:, 0, 1] = -v[..., 2, 0]
    out_array[:, 0, 2] = v[..., 1, 0]
    out_array[:, 1, 0] = v[..., 2, 0]
    out_array[:, 1, 2] = -v[..., 0, 0]
    out_array[:, 2, 0] = -v[..., 1, 0]
    out_array[:, 2, 1] = v[..., 0, 0]

    if v.ndim == 2:
        out_array = out_array.squeeze()

    return out_array


def twist_to_H_matrix(twist):
    """
    twist_to_H_matrix Convert twist coordinates to 4x4 homogeneous matrix
    Input: -twist(6,1): twist coordinates. Stack linear and angular parts [v;w]
    Output: -H(4,4): Euclidean transformation matrix (rigid body motion)
    """
    v = twist[:3]  # linear part
    w = twist[3:]  # angular part

    se_matrix = np.concatenate(
        [to_skew_symmetric_matrix(w.reshape(3, 1)), v[:, None]], axis=1
    )
    se_matrix = np.concatenate([se_matrix, np.zeros([1, 4])], axis=0)

    H = expm(se_matrix)

    return H


def H_matrix_to_twist(H):
    """
    H_matrix_to_twist Convert 4x4 homogeneous matrix to twist coordinates
    Input:
     -H(4,4): Euclidean transformation matrix (rigid body motion)
    Output:
     -twist(6,1): twist coordinates. Stack linear and angular parts [v;w]

    Observe that the same H might be represented by different twist vectors
    Here, twist(4:6) is a rotation vector with norm in [0,pi]
    """

    se_matrix = logm(H)

    # careful for rotations of pi; the top 3x3 submatrix of the returned se_matrix by logm is not
    # skew-symmetric (bad).

    v = se_matrix[:3, 3]

    w = skew_matrix_to_cross(se_matrix[:3, :3])
    twist = np.concatenate([v, w])

    return twist


def skew_matrix_to_cross(M):
    """
    Computes the 3D vector x corresponding to an antisymmetric matrix M such that M*y = cross(x,y)
    for all 3D vectors y.
    Input:
     - M(3,3) : antisymmetric matrix
    Output:
     - x(3,1) : column vector
    See also CROSS2MATRIX
    """
    x = np.array([-M[1, 2], M[0, 2], -M[0, 1]])

    return x
