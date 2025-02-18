from .utils import distance, majority_vote

class kNN():
    """
    Implementation of a k-nearest neighbors classifier.
    """
    def __init__(self, k: int):
        """
        Initialize the kNN classifier with the number of neighbors.

        :param int k: the number of neighbors to consider.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        self.k = k

    def _get_k_nearest_neighbors(
            self,
            X: list[list[float]],
            y: list[int],
            x: list[float]
        ) -> list[int]:
        """
        Return the labels of the k nearest neighbors of x.

        :param list[list[float]] X: the list of data points.
        :param list[int] y: the list of class labels.
        :param list[float] x: the new data point.
        """
        dist = [distance(x, point) for point in X]
        sorted_neighbors = sorted(enumerate(dist), key=lambda pair: pair[1])
        k_neighbors = [y[i] for i, _ in sorted_neighbors[:self.k]]
        return k_neighbors

    def __call__(
            self,
            data: tuple[list[list[float]], list[int]],
            new_points: list[list[float]]
        ):
        """
        Classify new points based on the training data.

        :param tuple[list[list[float]], list[int]] data: the training data.
        :param list[list[float]] new_points: the new data points to classify.
        """
        X, y = data
        self.prediction = [
            majority_vote(
                self._get_k_nearest_neighbors(X, y, point)
            )
            for point in new_points
        ]
