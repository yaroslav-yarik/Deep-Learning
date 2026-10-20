from .layers import *


def affine_relu_forward(x, w, b):
    """
    Составной слой («сэндвич»): аффинное преобразование, за которым
    следует ReLU.

    Входные данные:
    - x: вход аффинного слоя
    - w, b: веса и смещения аффинного слоя

    Возвращает кортеж:
    - out: выход ReLU
    - cache: объект, который нужно передать в обратный проход
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Обратный проход для составного слоя «аффинный слой + ReLU».
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

