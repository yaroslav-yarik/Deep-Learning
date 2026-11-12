import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Многослойная полносвязная нейронная сеть.

    Сеть содержит произвольное число скрытых слоёв, нелинейности ReLU и
    функцию потерь softmax. Кроме того, в сети можно (по желанию) использовать
    dropout (случайное отключение нейронов) и батч-нормализацию или
    нормализацию по слою. Для сети из L слоёв архитектура имеет вид

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    где батч-нормализация / нормализация по слою и dropout необязательны,
    а блок {...} повторяется L - 1 раз.

    Обучаемые параметры хранятся в словаре self.params и обучаются
    с помощью класса Solver.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Инициализирует новый объект FullyConnectedNet.

        Входные данные:
        - hidden_dims: список целых чисел — размеры всех скрытых слоёв.
        - input_dim: целое число, размер входа.
        - num_classes: целое число, количество классов.
        - dropout_keep_ratio: скаляр от 0 до 1, задающий силу dropout, —
            вероятность сохранить нейрон (чем меньше значение, тем сильнее
            dropout). Если dropout_keep_ratio=1, сеть вообще не должна
            использовать dropout.
        - normalization: тип нормализации, которую использует сеть. Допустимые
            значения: "batchnorm", "layernorm" или None — без нормализации
            (по умолчанию).
        - reg: скаляр, коэффициент L2-регуляризации.
        - weight_scale: скаляр, стандартное отклонение для случайной
            инициализации весов.
        - dtype: тип данных numpy; все вычисления будут выполняться в этом
            типе. float32 работает быстрее, но менее точен, поэтому для
            численной проверки градиента используйте float64.
        - seed: если не None, это зерно генератора случайных чисел (seed)
            передаётся слоям dropout. Тогда слои dropout работают
            детерминированно и для модели можно выполнить проверку градиента.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # Веса и смещения — как в лабораторной работе 1. Размеры входов/выходов
        # всех слоёв подряд: [input_dim, h_1, ..., h_{L-1}, num_classes];
        # слой i (нумерация с 1) отображает dims[i-1] -> dims[i].
        dims = [input_dim] + list(hidden_dims) + [num_classes]
        for i in range(1, self.num_layers + 1):
            self.params[f"W{i}"] = weight_scale * np.random.randn(dims[i - 1], dims[i])
            self.params[f"b{i}"] = np.zeros(dims[i])

        ############################################################################
        # TODO: При использовании нормализации (батч-нормализации или              #
        # нормализации по слою) храните параметры масштаба и сдвига первого слоя   #
        # в gamma1 и beta1, второго — в gamma2 и beta2 и т. д. У последнего слоя   #
        # нормализации нет. Параметры масштаба инициализируйте единицами, а        #
        # параметры сдвига — нулями.                                               #
        ############################################################################

        if self.normalization is not None:
            for i in range(1, self.num_layers):
                self.params[f"gamma{i}"] = np.ones(dims[i])
                self.params[f"beta{i}"] = np.zeros(dims[i])

        ############################################################################
        #                            КОНЕЦ ВАШЕГО КОДА                             #
        ############################################################################

        # При использовании dropout каждому слою dropout нужно передавать
        # словарь dropout_param, чтобы слой знал параметр p (вероятность
        # сохранить нейрон) и режим работы (train / test). Во все слои dropout
        # можно передавать один и тот же словарь dropout_param.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # При батч-нормализации нужно отслеживать скользящие средние и
        # скользящие дисперсии, поэтому каждому слою батч-нормализации нужно
        # передавать специальный словарь bn_param. В прямой проход первого слоя
        # батч-нормализации передавайте self.bn_params[0], в прямой проход
        # второго слоя — self.bn_params[1] и т. д.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Приводим все параметры к нужному типу данных.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Вычисляет значение функции потерь и градиент для полносвязной сети.

        Входные данные:
        - X: массив входных данных формы (N, d_1, ..., d_k)
        - y: массив меток формы (N,); y[i] — метка для X[i].

        Возвращает:
        Если y равен None, выполняет прямой проход модели в режиме тестирования
        и возвращает:
        - scores: массив формы (N, C) с оценками классов, где scores[i, c] —
            оценка класса c для X[i].

        Если y не равен None, выполняет прямой и обратный проходы в режиме
        обучения и возвращает кортеж из:
        - loss: скаляр, значение функции потерь
        - grads: словарь с теми же ключами, что и self.params, сопоставляющий
            именам параметров градиенты функции потерь по этим параметрам.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Устанавливаем режим train/test для параметров батч-нормализации и
        # dropout: при обучении и при тестировании эти слои ведут себя по-разному.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        # Прямой проход сети из лабораторной работы 1, разбитый на отдельные
        # слои, чтобы между ними можно было вставить нормализацию и dropout.
        L = self.num_layers
        caches = {}  # номер слоя -> кэши его частей для обратного прохода
        out = X
        for i in range(1, L):
            out, fc_cache = affine_forward(out, self.params[f"W{i}"], self.params[f"b{i}"])
            norm_cache, dropout_cache = None, None
            ########################################################################
            # TODO: Если используется нормализация (self.normalization равно       #
            # "batchnorm" или "layernorm"), примените её к out перед ReLU и        #
            # сохраните кэш в norm_cache. Слою i передавайте параметры gamma{i},   #
            # beta{i} и словарь self.bn_params[i - 1].                             #
            ########################################################################

            if self.normalization == "batchnorm":
                out, norm_cache = batchnorm_forward(
                    out, self.params[f"gamma{i}"], self.params[f"beta{i}"], self.bn_params[i - 1])
            elif self.normalization == "layernorm":
                out, norm_cache = layernorm_forward(
                    out, self.params[f"gamma{i}"], self.params[f"beta{i}"], self.bn_params[i - 1])

            ########################################################################
            #                          КОНЕЦ ВАШЕГО КОДА                           #
            ########################################################################
            out, relu_cache = relu_forward(out)
            ########################################################################
            # TODO: Если используется dropout (self.use_dropout), примените его к  #
            # out сразу после ReLU и сохраните кэш в dropout_cache. Во все слои    #
            # dropout передавайте словарь self.dropout_param.                      #
            ########################################################################

            if self.use_dropout:
                out, dropout_cache = dropout_forward(out, self.dropout_param)

            ########################################################################
            #                          КОНЕЦ ВАШЕГО КОДА                           #
            ########################################################################
            caches[i] = (fc_cache, norm_cache, relu_cache, dropout_cache)

        # Последний слой — только affine (softmax учитывается в функции потерь).
        scores, caches[L] = affine_forward(out, self.params[f"W{L}"], self.params[f"b{L}"])

        # В режиме тестирования сразу возвращаем оценки.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        # Обратный проход сети из лабораторной работы 1: потери на данных и
        # градиент по оценкам, затем слои в обратном порядке.
        loss, dout = softmax_loss(scores, y)
        dout, grads[f"W{L}"], grads[f"b{L}"] = affine_backward(dout, caches[L])
        for i in range(L - 1, 0, -1):
            fc_cache, norm_cache, relu_cache, dropout_cache = caches[i]
            ########################################################################
            # TODO: Если используется dropout, выполните обратный проход через     #
            # слой dropout (обновите dout).                                        #
            ########################################################################

            if self.use_dropout:
                dout = dropout_backward(dout, dropout_cache)

            ########################################################################
            #                          КОНЕЦ ВАШЕГО КОДА                           #
            ########################################################################
            dout = relu_backward(dout, relu_cache)
            ########################################################################
            # TODO: Если используется нормализация, выполните обратный проход      #
            # через слой нормализации (обновите dout) и сохраните градиенты        #
            # параметров масштаба и сдвига в grads[f"gamma{i}"] и                  #
            # grads[f"beta{i}"].                                                   #
            ########################################################################

            if self.normalization == "batchnorm":
                dout, grads[f"gamma{i}"], grads[f"beta{i}"] = batchnorm_backward_alt(dout, norm_cache)
            elif self.normalization == "layernorm":
                dout, grads[f"gamma{i}"], grads[f"beta{i}"] = layernorm_backward(dout, norm_cache)

            ########################################################################
            #                          КОНЕЦ ВАШЕГО КОДА                           #
            ########################################################################
            dout, grads[f"W{i}"], grads[f"b{i}"] = affine_backward(dout, fc_cache)

        # L2-регуляризация весов (смещения, а также параметры масштаба и сдвига
        # не регуляризуем): потери 0.5 * reg * ||W||^2, их градиент по W — reg * W.
        for i in range(1, L + 1):
            W = self.params[f"W{i}"]
            loss += 0.5 * self.reg * np.sum(W * W)
            grads[f"W{i}"] += self.reg * W

        return loss, grads
