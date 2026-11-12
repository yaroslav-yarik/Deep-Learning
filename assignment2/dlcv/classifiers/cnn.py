import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    Трёхслойная свёрточная сеть со следующей архитектурой:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    Сеть работает с мини-батчами данных формы (N, C, H, W), состоящими из
    N изображений высотой H и шириной W с C входными каналами.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Инициализирует новую сеть.

        Входные данные:
        - input_dim: кортеж (C, H, W), размер входных данных
        - num_filters: количество фильтров в свёрточном слое
        - filter_size: ширина (она же высота) фильтров свёрточного слоя
        - hidden_dim: количество нейронов в полносвязном скрытом слое
        - num_classes: количество оценок (scores) классов, которые выдаёт
          последний аффинный слой.
        - weight_scale: скаляр, стандартное отклонение для случайной
          инициализации весов.
        - reg: скаляр, коэффициент L2-регуляризации
        - dtype: тип данных numpy, в котором выполняются вычисления.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Инициализируйте веса и смещения трёхслойной свёрточной сети.       #
        # Веса инициализируйте из нормального (гауссова) распределения с центром   #
        # в 0.0 и стандартным отклонением weight_scale, а смещения — нулями.       #
        # Все веса и смещения храните в словаре self.params: веса и смещения       #
        # свёрточного слоя — под ключами 'W1' и 'b1', скрытого аффинного слоя —    #
        # под ключами 'W2' и 'b2', выходного аффинного слоя — под ключами          #
        # 'W3' и 'b3'.                                                             #
        #                                                                          #
        # ВАЖНО: в этой лабораторной работе можно считать, что дополнение          #
        # (padding) и шаг (stride) первого свёрточного слоя подобраны так, что     #
        # **ширина и высота входа сохраняются**. Посмотрите на начало метода       #
        # loss(), чтобы увидеть, как это сделано.                                  #
        ############################################################################

        C, H, W = input_dim
        self.params["W1"] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params["b1"] = np.zeros(num_filters)
        self.params["W2"] = weight_scale * np.random.randn(num_filters * (H // 2) * (W // 2), hidden_dim)
        self.params["b2"] = np.zeros(hidden_dim)
        self.params["W3"] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params["b3"] = np.zeros(num_classes)

        ############################################################################
        #                            КОНЕЦ ВАШЕГО КОДА                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Вычисляет значение функции потерь и градиент для трёхслойной свёрточной
        сети.

        Входные и выходные данные: тот же интерфейс (API), что у метода loss
        класса FullyConnectedNet в fc_net.py (и у TwoLayerNet из первой
        лабораторной работы).
        """
        X = X.astype(self.dtype)
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # conv_param передаётся в прямой проход свёрточного слоя.
        # Дополнение и шаг подобраны так, чтобы сохранить пространственный
        # размер входа.
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pool_param передаётся в прямой проход слоя max-пулинга.
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Реализуйте прямой проход трёхслойной свёрточной сети: вычислите    #
        # оценки классов для X и сохраните их в переменной scores.                 #
        #                                                                          #
        # Напомним, что в реализации можно использовать функции из                 #
        # dlcv/fast_layers.py и dlcv/layer_utils.py (они уже импортированы).       #
        ############################################################################

        pooled, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        hidden, hidden_cache = affine_relu_forward(pooled, W2, b2)
        scores, scores_cache = affine_forward(hidden, W3, b3)

        ############################################################################
        #                            КОНЕЦ ВАШЕГО КОДА                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Реализуйте обратный проход трёхслойной свёрточной сети. Сохраните  #
        # значение функции потерь в переменной loss, а градиенты — в словаре       #
        # grads. Потери на данных (data loss) вычисляйте с помощью softmax и       #
        # следите, чтобы grads[k] содержал градиенты для self.params[k].           #
        # Не забудьте добавить L2-регуляризацию!                                   #
        #                                                                          #
        # ПРИМЕЧАНИЕ: чтобы ваша реализация совпала с эталонной и прошла           #
        # проверки в ноутбуке, L2-регуляризация должна включать множитель 0.5      #
        # (он упрощает выражение для градиента).                                   #
        ############################################################################

        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * sum(np.sum(self.params[k] ** 2) for k in ("W1", "W2", "W3"))
        dhidden, grads["W3"], grads["b3"] = affine_backward(dscores, scores_cache)
        dpooled, grads["W2"], grads["b2"] = affine_relu_backward(dhidden, hidden_cache)
        _, grads["W1"], grads["b1"] = conv_relu_pool_backward(dpooled, conv_cache)
        for k in ("W1", "W2", "W3"):
            grads[k] += self.reg * self.params[k]

        ############################################################################
        #                            КОНЕЦ ВАШЕГО КОДА                             #
        ############################################################################

        return loss, grads
