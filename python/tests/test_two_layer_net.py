"""two_layer_net.ipynb — dlcv/layers.py и TwoLayerNet в dlcv/classifiers/fc_net.py."""
import numpy as np
import pytest

from dlcv.classifiers.fc_net import TwoLayerNet
from dlcv.layers import (
    affine_backward,
    affine_forward,
    relu_backward,
    relu_forward,
    softmax_loss,
)
from dlcv.layer_utils import affine_relu_backward, affine_relu_forward
from tests.utils import eval_numerical_gradient, eval_numerical_gradient_array, rel_error


def test_affine_forward():
    x = np.linspace(-0.1, 0.5, num=240).reshape(2, 4, 5, 6)
    w = np.linspace(-0.2, 0.3, num=360).reshape(120, 3)
    b = np.linspace(-0.3, 0.1, num=3)
    out, _ = affine_forward(x, w, b)
    correct = np.array([[1.49834967, 1.70660132, 1.91485297],
                        [3.25553199, 3.5141327, 3.77273342]])
    assert rel_error(out, correct) < 1e-8, "affine_forward: неверный выход"


def test_affine_backward():
    np.random.seed(231)
    x, w, b = np.random.randn(10, 2, 3), np.random.randn(6, 5), np.random.randn(5)
    dout = np.random.randn(10, 5)
    _, cache = affine_forward(x, w, b)
    dx, dw, db = affine_backward(dout, cache)
    assert rel_error(eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout), dx) < 1e-9, "dx"
    assert rel_error(eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout), dw) < 1e-9, "dw"
    assert rel_error(eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout), db) < 1e-9, "db"


def test_relu_forward():
    out, _ = relu_forward(np.linspace(-0.5, 0.5, num=12).reshape(3, 4))
    correct = np.array([[0., 0., 0., 0.],
                        [0., 0., 0.04545455, 0.13636364],
                        [0.22727273, 0.31818182, 0.40909091, 0.5]])
    assert rel_error(out, correct) < 1e-7, "relu_forward: неверный выход"


def test_relu_backward():
    np.random.seed(231)
    x = np.random.randn(10, 10)
    dout = np.random.randn(*x.shape)
    _, cache = relu_forward(x)
    numeric = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)
    assert rel_error(numeric, relu_backward(dout, cache)) < 1e-10, "relu_backward: неверный градиент"


def test_affine_relu_sandwich():
    np.random.seed(231)
    x, w, b = np.random.randn(2, 3, 4), np.random.randn(12, 10), np.random.randn(10)
    dout = np.random.randn(2, 10)
    _, cache = affine_relu_forward(x, w, b)
    dx, dw, db = affine_relu_backward(dout, cache)
    assert rel_error(eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, dout), dx) < 1e-9
    assert rel_error(eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, dout), dw) < 1e-9
    assert rel_error(eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, dout), db) < 1e-9


def test_softmax_loss():
    np.random.seed(231)
    x = 0.001 * np.random.randn(50, 10)
    y = np.random.randint(10, size=50)
    loss, dx = softmax_loss(x, y)
    assert abs(loss - 2.3) < 0.01, "потери при маленьких оценках должны быть ≈ 2.3"
    numeric = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
    assert rel_error(numeric, dx) < 1e-7, "softmax_loss: неверный градиент"


@pytest.fixture
def two_layer():
    np.random.seed(231)
    N, D, H, C = 3, 5, 50, 7
    model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=1e-3)
    return model, N, D, H, C


def test_two_layer_init(two_layer):
    model, *_ = two_layer
    assert abs(model.params["W1"].std() - 1e-3) < 1e-4, "W1: неверная инициализация"
    assert abs(model.params["W2"].std() - 1e-3) < 1e-4, "W2: неверная инициализация"
    assert np.all(model.params["b1"] == 0) and np.all(model.params["b2"] == 0), "смещения должны быть нулями"


def _fixed(model, N, D, H, C):
    model.params["W1"] = np.linspace(-0.7, 0.3, num=D * H).reshape(D, H)
    model.params["b1"] = np.linspace(-0.1, 0.9, num=H)
    model.params["W2"] = np.linspace(-0.3, 0.4, num=H * C).reshape(H, C)
    model.params["b2"] = np.linspace(-0.9, 0.1, num=C)
    return np.linspace(-5.5, 4.5, num=N * D).reshape(D, N).T


def test_two_layer_scores_and_loss(two_layer):
    model, N, D, H, C = two_layer
    X = _fixed(model, N, D, H, C)
    correct_scores = np.asarray(
        [[11.53165108, 12.2917344, 13.05181771, 13.81190102, 14.57198434, 15.33206765, 16.09215096],
         [12.05769098, 12.74614105, 13.43459113, 14.1230412, 14.81149128, 15.49994135, 16.18839143],
         [12.58373087, 13.20054771, 13.81736455, 14.43418138, 15.05099822, 15.66781506, 16.2846319]])
    assert np.abs(model.loss(X) - correct_scores).sum() < 1e-6, "неверный прямой проход"
    y = np.asarray([0, 5, 1])
    loss, _ = model.loss(X, y)
    assert abs(loss - 3.4702243556) < 1e-10, "неверные потери без регуляризации"
    model.reg = 1.0
    loss, _ = model.loss(X, y)
    assert abs(loss - 26.5948426952) < 1e-10, "неверные потери с регуляризацией"


@pytest.mark.parametrize("reg", [0.0, 0.7])
def test_two_layer_gradients(two_layer, reg):
    model, N, D, H, C = two_layer
    X = _fixed(model, N, D, H, C)
    y = np.asarray([0, 5, 1])
    model.reg = reg
    _, grads = model.loss(X, y)
    assert sorted(grads) == ["W1", "W2", "b1", "b2"]
    for name in sorted(grads):
        numeric = eval_numerical_gradient(lambda _: model.loss(X, y)[0], model.params[name], verbose=False)
        assert rel_error(numeric, grads[name]) < 1e-6, f"градиент {name} не совпадает с численным"
