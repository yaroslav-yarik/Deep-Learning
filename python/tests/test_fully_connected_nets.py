"""FullyConnectedNets.ipynb — FullyConnectedNet в dlcv/classifiers/fc_net.py и dlcv/optim.py."""
import numpy as np
import pytest

from dlcv.classifiers.fc_net import FullyConnectedNet
from dlcv.optim import adam, rmsprop, sgd_momentum
from tests.utils import eval_numerical_gradient, rel_error


def test_init_shapes_and_scale():
    np.random.seed(0)
    model = FullyConnectedNet([100, 80], input_dim=60, num_classes=10, weight_scale=5e-2)
    shapes = {name: value.shape for name, value in model.params.items()}
    assert shapes == {"W1": (60, 100), "b1": (100,), "W2": (100, 80), "b2": (80,),
                      "W3": (80, 10), "b3": (10,)}, "неверные имена или формы параметров"
    assert abs(model.params["W2"].std() - 5e-2) < 5e-3, "веса должны иметь std = weight_scale"
    assert all(np.all(model.params[f"b{i}"] == 0) for i in (1, 2, 3)), "смещения должны быть нулями"


def test_test_mode_returns_scores():
    np.random.seed(0)
    model = FullyConnectedNet([7], input_dim=4, num_classes=3, dtype=np.float64)
    scores = model.loss(np.random.randn(5, 4))
    assert scores.shape == (5, 3), "без y должен возвращаться массив оценок (N, C)"


@pytest.mark.parametrize("reg", [0.0, 3.14])
def test_gradients(reg):
    np.random.seed(231)
    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))
    model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C, reg=reg,
                              weight_scale=5e-2, dtype=np.float64)
    loss, grads = model.loss(X, y)
    assert np.isfinite(loss)
    names = ["W1", "W2", "W3", "b1", "b2", "b3"]
    assert sorted(model.params) == names and sorted(grads) == names, \
        "нужны параметры и градиенты W1..W3, b1..b3"
    for name in sorted(grads):
        numeric = eval_numerical_gradient(lambda _: model.loss(X, y)[0], model.params[name],
                                          verbose=False, h=1e-5)
        # По ноутбуку: ~1e-7, при reg = 0 для W2 допустимо ~1e-5.
        assert rel_error(numeric, grads[name]) < 1e-4, f"градиент {name} не совпадает с численным"


def _state(N=4, D=5):
    w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
    return w, dw, N, D


def test_sgd_momentum():
    w, dw, N, D = _state()
    config = {"learning_rate": 1e-3, "velocity": np.linspace(0.6, 0.9, num=N * D).reshape(N, D)}
    next_w, _ = sgd_momentum(w, dw, config=config)
    expected_next_w = np.asarray([
        [0.1406, 0.20738947, 0.27417895, 0.34096842, 0.40775789],
        [0.47454737, 0.54133684, 0.60812632, 0.67491579, 0.74170526],
        [0.80849474, 0.87528421, 0.94207368, 1.00886316, 1.07565263],
        [1.14244211, 1.20923158, 1.27602105, 1.34281053, 1.4096]])
    expected_velocity = np.asarray([
        [0.5406, 0.55475789, 0.56891579, 0.58307368, 0.59723158],
        [0.61138947, 0.62554737, 0.63970526, 0.65386316, 0.66802105],
        [0.68217895, 0.69633684, 0.71049474, 0.72465263, 0.73881053],
        [0.75296842, 0.76712632, 0.78128421, 0.79544211, 0.8096]])
    assert rel_error(next_w, expected_next_w) < 1e-6, "sgd_momentum: неверный next_w"
    assert rel_error(config["velocity"], expected_velocity) < 1e-6, "sgd_momentum: неверная velocity"


def test_rmsprop():
    w, dw, N, D = _state()
    config = {"learning_rate": 1e-2, "cache": np.linspace(0.6, 0.9, num=N * D).reshape(N, D)}
    next_w, _ = rmsprop(w, dw, config=config)
    expected_next_w = np.asarray([
        [-0.39223849, -0.34037513, -0.28849239, -0.23659121, -0.18467247],
        [-0.132737, -0.08078555, -0.02881884, 0.02316247, 0.07515774],
        [0.12716641, 0.17918792, 0.23122175, 0.28326742, 0.33532447],
        [0.38739248, 0.43947102, 0.49155973, 0.54365823, 0.59576619]])
    expected_cache = np.asarray([
        [0.5976, 0.6126277, 0.6277108, 0.64284931, 0.65804321],
        [0.67329252, 0.68859723, 0.70395734, 0.71937285, 0.73484377],
        [0.75037008, 0.7659518, 0.78158892, 0.79728144, 0.81302936],
        [0.82883269, 0.84469141, 0.86060554, 0.87657507, 0.8926]])
    assert rel_error(next_w, expected_next_w) < 1e-6, "rmsprop: неверный next_w"
    assert rel_error(config["cache"], expected_cache) < 1e-6, "rmsprop: неверный cache"


def test_adam():
    w, dw, N, D = _state()
    config = {"learning_rate": 1e-2,
              "m": np.linspace(0.6, 0.9, num=N * D).reshape(N, D),
              "v": np.linspace(0.7, 0.5, num=N * D).reshape(N, D), "t": 5}
    next_w, _ = adam(w, dw, config=config)
    expected_next_w = np.asarray([
        [-0.40094747, -0.34836187, -0.29577703, -0.24319299, -0.19060977],
        [-0.1380274, -0.08544591, -0.03286534, 0.01971428, 0.0722929],
        [0.1248705, 0.17744702, 0.23002243, 0.28259667, 0.33516969],
        [0.38774145, 0.44031188, 0.49288093, 0.54544852, 0.59801459]])
    expected_v = np.asarray([
        [0.69966, 0.68908382, 0.67851319, 0.66794809, 0.65738853],
        [0.64683452, 0.63628604, 0.6257431, 0.61520571, 0.60467385],
        [0.59414753, 0.58362676, 0.57311152, 0.56260183, 0.55209767],
        [0.54159906, 0.53110598, 0.52061845, 0.51013645, 0.49966]])
    expected_m = np.asarray([
        [0.48, 0.49947368, 0.51894737, 0.53842105, 0.55789474],
        [0.57736842, 0.59684211, 0.61631579, 0.63578947, 0.65526316],
        [0.67473684, 0.69421053, 0.71368421, 0.73315789, 0.75263158],
        [0.77210526, 0.79157895, 0.81105263, 0.83052632, 0.85]])
    assert rel_error(next_w, expected_next_w) < 1e-6, "adam: неверный next_w"
    assert rel_error(config["v"], expected_v) < 1e-6, "adam: неверный v"
    assert rel_error(config["m"], expected_m) < 1e-6, "adam: неверный m"
