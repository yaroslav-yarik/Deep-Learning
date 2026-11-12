import numpy as np

"""
В этом файле реализованы различные правила обновления первого порядка, которые
часто применяются при обучении нейронных сетей. Каждое правило обновления
принимает текущие веса и градиент функции потерь по этим весам и вычисляет
следующий набор весов. У всех правил обновления одинаковый интерфейс:

def update(w, dw, config=None):

Входные данные:
  - w: массив numpy с текущими весами.
  - dw: массив numpy той же формы, что и w, — градиент функции потерь
    по w.
  - config: словарь со значениями гиперпараметров, таких как скорость
    обучения (learning rate), момент (momentum) и т. д. Если правилу
    обновления нужно сохранять (кэшировать) какие-то значения между
    итерациями, то config хранит и эти значения.

Возвращает:
  - next_w: следующая точка после обновления.
  - config: словарь config, который нужно передать в следующую итерацию
    правила обновления.

ПРИМЕЧАНИЕ: для большинства правил обновления скорость обучения по умолчанию,
скорее всего, окажется неудачной; однако значения по умолчанию остальных
гиперпараметров должны хорошо работать для самых разных задач.

Ради эффективности правила обновления могут выполнять обновление на месте
(in-place), изменяя w и присваивая next_w = w.
"""


def sgd(w, dw, config=None):
    """
    Выполняет обычный (vanilla) стохастический градиентный спуск.

    Формат config:
    - learning_rate: скаляр, скорость обучения.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Выполняет стохастический градиентный спуск с моментом (momentum).

    Формат config:
    - learning_rate: скаляр, скорость обучения.
    - momentum: скаляр от 0 до 1 — значение момента.
      При momentum = 0 метод сводится к обычному SGD.
    - velocity: массив numpy той же формы, что w и dw, в котором хранится
      «скорость» (velocity) — затухающая сумма предыдущих шагов обновления.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    # Реализация из лабораторной работы 1.
    # Скорость: затухающая сумма прошлых шагов плюс шаг против градиента.
    v = config["momentum"] * v - config["learning_rate"] * dw
    # Сдвигаем веса на величину скорости.
    next_w = w + v
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Правило обновления RMSProp: использует скользящее среднее квадратов
    градиентов, чтобы задать адаптивную скорость обучения для каждого
    параметра.

    Формат config:
    - learning_rate: скаляр, скорость обучения.
    - decay_rate: скаляр от 0 до 1 — коэффициент затухания для кэша
      квадратов градиентов.
    - epsilon: малая константа для сглаживания, чтобы избежать деления на ноль.
    - cache: скользящее среднее вторых моментов градиентов.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    # Реализация из лабораторной работы 1.
    # Скользящее среднее квадратов градиента (поэлементно).
    decay = config["decay_rate"]
    config["cache"] = decay * config["cache"] + (1 - decay) * dw**2
    # Шаг нормируется на корень из среднего квадрата градиента:
    # у каждого параметра получается своя эффективная скорость обучения.
    next_w = w - config["learning_rate"] * dw / (
        np.sqrt(config["cache"]) + config["epsilon"]
    )

    return next_w, config


def adam(w, dw, config=None):
    """
    Правило обновления Adam: использует скользящие средние как самого
    градиента, так и его квадрата, а также коррекцию смещения
    (bias correction).

    Формат config:
    - learning_rate: скаляр, скорость обучения.
    - beta1: коэффициент затухания скользящего среднего первого момента
      градиента.
    - beta2: коэффициент затухания скользящего среднего второго момента
      градиента.
    - epsilon: малая константа для сглаживания, чтобы избежать деления на ноль.
    - m: скользящее среднее градиента.
    - v: скользящее среднее квадрата градиента.
    - t: номер итерации.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    # Реализация из лабораторной работы 1.
    beta1, beta2 = config["beta1"], config["beta2"]

    # Номер итерации увеличиваем до его использования (нумерация с 1).
    config["t"] += 1
    t = config["t"]

    # Скользящие средние градиента (первый момент) и его квадрата (второй).
    config["m"] = beta1 * config["m"] + (1 - beta1) * dw
    config["v"] = beta2 * config["v"] + (1 - beta2) * dw**2

    # Коррекция смещения: m и v инициализированы нулями и на первых
    # итерациях занижены; делим на (1 - beta^t), чтобы это компенсировать.
    m_hat = config["m"] / (1 - beta1**t)
    v_hat = config["v"] / (1 - beta2**t)

    # Шаг в направлении сглаженного градиента с поэлементной нормировкой.
    next_w = w - config["learning_rate"] * m_hat / (
        np.sqrt(v_hat) + config["epsilon"]
    )

    return next_w, config
