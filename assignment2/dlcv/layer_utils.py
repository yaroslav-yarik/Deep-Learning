from .layers import *
from .fast_layers import *


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


def conv_relu_forward(x, w, b, conv_param):
    """
    Составной слой («сэндвич»): свёртка, за которой следует ReLU.

    Входные данные:
    - x: вход свёрточного слоя
    - w, b, conv_param: веса и параметры свёрточного слоя

    Возвращает кортеж:
    - out: выход ReLU
    - cache: объект, который нужно передать в обратный проход
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Обратный проход для составного слоя «свёртка + ReLU».
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Составной слой («сэндвич»): свёртка, затем ReLU и пулинг.

    Входные данные:
    - x: вход свёрточного слоя
    - w, b, conv_param: веса и параметры свёрточного слоя
    - pool_param: параметры слоя пулинга

    Возвращает кортеж:
    - out: выход слоя пулинга
    - cache: объект, который нужно передать в обратный проход
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Обратный проход для составного слоя «свёртка + ReLU + пулинг».
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
