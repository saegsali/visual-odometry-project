import numpy as np
import pytest
from vo.algorithms.ransac import RANSAC

rng = np.random.default_rng(2023)


@pytest.fixture
def data() -> tuple:
    # Taken from main.py in exercise 07
    num_inliers = 20
    num_outliers = 10
    noise_ratio = 0.1
    poly = rng.uniform(size=[3, 1])  # random second-order polynomial
    extremum = -poly[1] / (2 * poly[0])
    xstart = extremum - 0.5
    lowest = np.polyval(poly, extremum)
    highest = np.polyval(poly, xstart)
    yspan = highest - lowest
    max_noise = noise_ratio * yspan
    x = rng.uniform(size=[1, num_inliers]) + xstart
    y = np.polyval(poly, x)
    y = y + (rng.uniform(size=y.shape) - 0.5) * 2 * max_noise
    data_array = np.concatenate(
        [
            np.concatenate([x, rng.uniform(size=[1, num_outliers]) + xstart], axis=1),
            np.concatenate(
                [y, rng.uniform(size=[1, num_outliers]) * yspan + lowest], axis=1
            ),
        ],
        axis=0,
    ).T

    return data_array, poly, max_noise


def model_fn(samples):
    # Parabola fitting
    p = np.polyfit(samples[:, 0], samples[:, 1], 2)
    return p


def error_fn(p, points):
    # Dummy error function for testing
    errors = np.abs(np.polyval(p, points[:, 0]) - points[:, 1])
    return errors


def test_ransac_find_best_model(data):
    population, true_model, max_noise = data
    inlier_threshold = max_noise + 1e-5
    outlier_ratio = 1 / 3
    confidence = 0.99

    ransac = RANSAC(
        3,
        population,
        model_fn,
        error_fn,
        inlier_threshold,
        outlier_ratio,
        confidence,
    )

    best_model, best_inliers = ransac.find_best_model()

    x = np.linspace(population[:, 0].min(), population[:, 0].max(), 100)
    y_true = np.polyval(true_model, x)
    y_guess = np.polyval(best_model, x)

    # Random seed is fixed, so we can check with absolute tolerance
    assert np.allclose(y_true, y_guess, atol=2e-3)


if __name__ == "__main__":
    pytest.main(["-v"])
