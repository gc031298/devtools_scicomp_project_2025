import os
import yaml

def distance(point1: list[float], point2: list[float]) -> float:
    """
    Compute the square of the Euclidean distance between two points.

    :param list[float] point1: the first point.
    :param list[float] point2: the second point.
    """
    assert len(point1) == len(point2), 'The points must have the same length.'
    return sum((p1 - p2)**2 for p1, p2 in zip(point1, point2))


def majority_vote(neighbors: list[int]) -> int:
    """
    Return the most common class among the neighbors.

    :param list[int] neighbors: the list of class labels.
    """
    return max(set(neighbors), key=neighbors.count)


def read_config(file: str) -> dict:
    """
    Read a YAML configuration file.

    :param str file: the name of the file.
    """
    filepath = os.path.abspath(f'{file}.yaml')
    with open(filepath, 'r') as stream:
        kwargs = yaml.safe_load(stream)
    return kwargs


def read_file(file: str) -> tuple[list[list[float]], list[int]]:
    """
    Read the dataset file.

    :param str file: the name of the file.
    """
    filepath = os.path.abspath(f'./{file}')
    features, labels = [], []
    with open(filepath, 'r') as stream:
        for line in stream:
            data = line.split(',')
            features.append([float(x) for x in data[:-1]])
            labels.append(0 if data[-1][0] == 'b' else 1)
    return features, labels
