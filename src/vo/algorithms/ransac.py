"""Basic RANSAC implementation for model estimation.
Takes a population of points and estimates a model using RANSAC, by sampling s points, computing the model 
and then computing the error for all points. If the error is below a threshold, the point is considered an inlier.
This process is repeated for a number of iterations and the model with the most inliers is returned. The model is
at the end recomputed using all inliers.

Example:
    See below for example usage or run with `python ransac.py`
"""
from typing import Callable

import numpy as np


class RANSAC:
    def __init__(
        self,
        s_points: int,
        population: np.ndarray | list | tuple,
        model_fn: Callable,
        error_fn: Callable,
        inlier_threshold: float,
        outlier_ratio: float = 0.9,
        confidence: float = 0.99,
        max_iterations: int = np.inf,
        adaptive: bool = True,
    ) -> None:
        """Initializes a RANSAC estimator for a provided model and error function.

        Args:
            s_points (int): Number of points to be sampled for model estimation.
            population (np.ndarray | list | tuple): The population of points to sample from for model estimation, indexed in first dimension.
            model_fn (function): Function which takes as input the sampled points and estimates the mdoel.
            error_fn (function): Function which computes the error terms for all points.
            inlier_threshold (float): The threshold for error term to be considered an inlier.
            outlier_ratio (float, optional): Percentage of estimated outlier points, serves as upper bound for adaptive case. Defaults to 0.9.
            confidence (float, optional): Confidence level that all inliers are selected. Defaults to 0.99.
            max_iterations (int, optional): Max iterations algorithm should perform before stopping. Defaults to np.inf.
            adaptive (bool, optional): Whether to use adaptive RANSAC by adjusting the number of iterations required using estimation of the outlier ratio. Defaults to True.
        """
        self.s = s_points
        self.population = np.array(population)
        self.model_fn = model_fn
        self.error_fn = error_fn
        self.inlier_threshold = inlier_threshold
        self.outlier_ratio = outlier_ratio
        self.confidence = confidence
        self.adaptive = adaptive

        self.rng = np.random.default_rng(2023)

        # Store number of iterations to perform
        self.max_iterations = max_iterations
        self.n_iterations = min(max_iterations, self.compute_n_iterations())

    def compute_n_iterations(self) -> int:
        """Computes the number of iterations to perform.

        Returns:
            int: Number of iterations considering the provided confidence level and outlier ratio.
        """
        k = np.ceil(
            np.log(1 - self.confidence) / np.log(1 - (1 - self.outlier_ratio) ** self.s)
        )
        return int(k)

    def find_best_model(self) -> tuple[object, np.ndarray]:
        """Finds the best model using RANSAC.

        Returns:
            object: The best model parameters or object.
            np.ndarray: Boolean array of inliers.
        """
        best_n_inliers = -1
        best_inliers = None
        n = 0

        while n < self.n_iterations:
            # Sample s points
            idxs = self.rng.choice(
                np.arange(len(self.population)), replace=False, size=self.s
            )
            points = self.population[idxs]

            # Compute model from sampled points
            model = self.model_fn(points)

            # Compute inliers
            errors = self.error_fn(model, self.population)
            inliers = errors < self.inlier_threshold
            n_inliers = inliers.sum()

            if n_inliers > best_n_inliers:
                best_n_inliers = n_inliers
                best_inliers = inliers

                # Adjust number of iterations if adaptive
                if self.adaptive:
                    self.outlier_ratio = 1 - best_n_inliers / len(self.population)
                    self.outlier_ratio = min(
                        max(self.outlier_ratio, 0.01), 0.99
                    )  # for numerical safety
                    num_iterations = self.compute_n_iterations()
                    self.n_iterations = int(min(self.max_iterations, num_iterations))
            n += 1

        # Recompute best model using all inliers
        all_inlier_points = self.population[best_inliers]
        best_model = self.model_fn(all_inlier_points)

        return best_model, best_inliers


if __name__ == "__main__":
    # Example usage
    def model_fn(points):
        # Parabola fitting
        p = np.polyfit(points[:, 0], points[:, 1], 2)
        return p

    def error_fn(p, points):
        # Dummy error function for testing
        errors = np.abs(np.polyval(p, points[:, 0]) - points[:, 1])
        return errors

    population = np.array(
        [[1, 1], [2, 5], [3, 8], [4, 16.5], [5, 23.75]]
    )  # N points of (x,y) coordintes
    inlier_threshold = 0.1
    outlier_ratio = 1 / 3
    confidence = 0.99
    max_iterations = 100

    ransac = RANSAC(
        3,
        population,
        model_fn,
        error_fn,
        inlier_threshold,
        outlier_ratio,
        confidence=confidence,
    )

    best_model, best_inliers = ransac.find_best_model()
    print(best_model)

    x = np.linspace(population[:, 0].min(), population[:, 0].max(), 100)
    y_guess = np.polyval(best_model, x)

    import matplotlib.pyplot as plt

    plt.plot(x, y_guess, label="Guess")
    plt.scatter(population[:, 0], population[:, 1], label="Data")
    plt.legend()
    plt.show()
