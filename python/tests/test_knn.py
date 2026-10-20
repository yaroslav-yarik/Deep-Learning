"""knn.ipynb — dlcv/classifiers/k_nearest_neighbor.py."""
import numpy as np
import pytest

from dlcv.classifiers import KNearestNeighbor


@pytest.fixture
def knn():
    rng = np.random.default_rng(231)
    classifier = KNearestNeighbor()
    classifier.train(rng.normal(size=(40, 12)), rng.integers(0, 5, size=40))
    X_test = rng.normal(size=(7, 12))
    expected = np.sqrt(((X_test[:, None, :] - classifier.X_train[None]) ** 2).sum(-1))
    return classifier, X_test, expected


def test_distances_two_loops(knn):
    classifier, X_test, expected = knn
    dists = classifier.compute_distances_two_loops(X_test)
    assert dists.shape == (7, 40), "матрица расстояний должна иметь форму (num_test, num_train)"
    assert np.allclose(dists, expected), "L2-расстояния посчитаны неверно"


def test_distances_one_loop(knn):
    classifier, X_test, expected = knn
    assert np.allclose(classifier.compute_distances_one_loop(X_test), expected), \
        "версия с одним циклом не совпадает с эталонными расстояниями"


def test_distances_no_loops(knn):
    classifier, X_test, expected = knn
    assert np.allclose(classifier.compute_distances_no_loops(X_test), expected), \
        "версия без циклов не совпадает с эталонными расстояниями"


def test_predict_labels_majority_and_ties():
    classifier = KNearestNeighbor()
    classifier.train(np.zeros((5, 1)), np.array([2, 1, 1, 0, 3]))
    dists = np.array([
        [0.1, 0.2, 0.3, 0.9, 0.8],   # k=3: соседи 2, 1, 1 → 1
        [0.9, 0.8, 0.7, 0.1, 0.2],   # k=3: соседи 0, 3, 1 → ничья → меньшая метка 0
    ])
    assert list(classifier.predict_labels(dists, k=1)) == [2, 0], "k=1: метка ближайшего соседа"
    assert list(classifier.predict_labels(dists, k=3)) == [1, 0], \
        "k=3: большинство голосов, при ничьей — меньшая метка"


def test_predict_matches_all_distance_versions(knn):
    classifier, X_test, expected = knn
    # Эталон: k ближайших и голосование (bincount.argmax при ничьей берёт меньшую метку).
    nearest = classifier.y_train[np.argsort(expected, axis=1, kind="stable")[:, :3]]
    reference = np.array([np.bincount(row).argmax() for row in nearest])
    for loops in (0, 1, 2):
        assert np.array_equal(classifier.predict(X_test, k=3, num_loops=loops), reference), \
            f"predict(num_loops={loops}) даёт другие метки"
