import pytest
from pyclassify.utils import distance, majority_vote
from pyclassify.classifier import kNN

def test_distance():
    """
    Test the distance properties.
    """
    # Define three points
    point1 = [0, 0, 0]
    point2 = [3, 4, 5]
    point3 = [-2, 4, 8]

    # Test the distance between points
    assert distance(point1, point2) == 50
    assert distance(point2, point3) == 34
    assert distance(point1, point3) == 84

    # Test the distance between a point and itself
    assert distance(point1, point1) == 0
    assert distance(point2, point2) == 0
    assert distance(point3, point3) == 0

    # Test the distance is symmetric
    assert distance(point1, point2) == distance(point2, point1)
    assert distance(point2, point3) == distance(point3, point2)
    assert distance(point1, point3) == distance(point3, point1)

    # Test the distance is positive
    assert distance(point1, point2) >= 0
    assert distance(point2, point3) >= 0
    assert distance(point1, point3) >= 0

    # Test the triangle inequality
    assert (
        distance(point1, point3) <=
        distance(point1, point2) +
        distance(point2, point3)
    )


def test_majority_vote():
    """
    Test the majority vote algorithm.
    """
    assert majority_vote([1, 0, 0, 0]) == 0
    assert majority_vote([1, 1, 1, 0]) == 1


def test_kNN_constructor():
    """
    Test the constructor of the kNN
    """
    # Test valid types
    kNN(5)

    # Test invalid types
    with pytest.raises(ValueError):
        kNN(0)
    with pytest.raises(ValueError):
        kNN(-1)
    with pytest.raises(ValueError):
        kNN(1.5)
    with pytest.raises(ValueError):
        kNN("1")
    with pytest.raises(ValueError):
        kNN([1])
