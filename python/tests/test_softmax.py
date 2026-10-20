"""softmax.ipynb — dlcv/classifiers/softmax.py и linear_classifier.py."""
import numpy as np
import pytest

from dlcv.classifiers import Softmax
from dlcv.classifiers.softmax import softmax_loss_naive, softmax_loss_vectorized
from tests.utils import eval_numerical_gradient, rel_error


@pytest.fixture
def batch():
    rng = np.random.default_rng(231)
    X = rng.normal(size=(30, 16))
    y = rng.integers(0, 10, size=30)
    W = rng.normal(size=(16, 10)) * 1e-3
    return W, X, y


@pytest.mark.parametrize("reg", [0.0, 50.0])
def test_naive_gradient(batch, reg):
    W, X, y = batch
    _, grad = softmax_loss_naive(W, X, y, reg)
    numeric = eval_numerical_gradient(lambda w: softmax_loss_naive(w, X, y, reg)[0], W, verbose=False)
    assert rel_error(numeric, grad) < 1e-6, "аналитический градиент не совпадает с численным"


@pytest.mark.parametrize("reg", [0.0, 5e-2])
def test_vectorized_matches_naive(batch, reg):
    W, X, y = batch
    loss_naive, grad_naive = softmax_loss_naive(W, X, y, reg)
    loss_vec, grad_vec = softmax_loss_vectorized(W, X, y, reg)
    assert abs(loss_naive - loss_vec) < 1e-10, "потери векторизованной версии отличаются"
    assert np.linalg.norm(grad_naive - grad_vec) < 1e-10, "градиент векторизованной версии отличается"


def test_predict_is_argmax_of_scores():
    rng = np.random.default_rng(0)
    model = Softmax()
    model.W = rng.normal(size=(6, 4))
    X = rng.normal(size=(9, 6))
    pred = model.predict(X)
    assert pred.shape == (9,)
    assert np.array_equal(pred, np.argmax(X @ model.W, axis=1)), "predict должен брать argmax оценок"


def test_sgd_training_reduces_loss():
    rng = np.random.default_rng(1)
    centers = rng.normal(scale=4, size=(3, 5))
    y = rng.integers(0, 3, size=300)
    X = centers[y] + rng.normal(size=(300, 5))
    np.random.seed(1)
    model = Softmax()
    history = model.train(X, y, learning_rate=1e-1, reg=1e-4, num_iters=150, batch_size=32)
    assert len(history) == 150, "train должен вернуть потери на каждой итерации"
    assert np.mean(history[-10:]) < 0.5 * history[0], "SGD не уменьшает потери"
    assert np.mean(model.predict(X) == y) > 0.9, "на хорошо разделимых данных точность должна быть > 0.9"
