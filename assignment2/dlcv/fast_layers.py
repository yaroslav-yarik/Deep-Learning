"""
Быстрые реализации слоёв свёртки и пулинга.

Всё здесь — векторизованный NumPy: компилировать расширения не нужно.
"""
import numpy as np

from .im2col import col2im_indices, im2col_indices


def col2im_6d(cols, N, C, H, W, HH, WW, pad, stride):
    """
    Операция, обратная к im2col через шаги по памяти (strides) из
    conv_forward_strides.

    Накапливает (суммирует) столбцы формы (C, HH, WW, N, out_h, out_w)
    обратно в массив формы (N, C, H, W). Циклы идут только по HH * WW
    позициям внутри фильтра; каждый шаг — векторизованная операция над всем
    мини-батчем.
    """
    out_h = (H + 2 * pad - HH) // stride + 1
    out_w = (W + 2 * pad - WW) // stride + 1
    x_padded = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)
    for hh in range(HH):
        for ww in range(WW):
            x_padded[
                :, :, hh : hh + stride * out_h : stride, ww : ww + stride * out_w : stride
            ] += cols[:, hh, ww].transpose(1, 0, 2, 3)
    if pad > 0:
        return x_padded[:, :, pad:-pad, pad:-pad]
    return x_padded


def conv_forward_im2col(x, w, b, conv_param):
    """
    Быстрая реализация прямого прохода для свёрточного слоя на основе
    im2col и col2im.
    """
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param["stride"], conv_param["pad"]

    # Проверяем размеры
    assert (W + 2 * pad - filter_width) % stride == 0, "ширина не подходит"
    assert (H + 2 * pad - filter_height) % stride == 0, "высота не подходит"

    # Создаём выходной массив
    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache


def conv_forward_strides(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param["stride"], conv_param["pad"]

    # Проверяем размеры
    # assert (W + 2 * pad - WW) % stride == 0, 'ширина не подходит'
    # assert (H + 2 * pad - HH) % stride == 0, 'высота не подходит'

    # Дополняем вход нулями
    p = pad
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    # Вычисляем размеры выхода
    H += 2 * pad
    W += 2 * pad
    out_h = (H - HH) // stride + 1
    out_w = (W - WW) // stride + 1

    # Выполняем im2col, хитро подобрав шаги по памяти (strides)
    shape = (C, HH, WW, N, out_h, out_w)
    strides = (H * W, W, 1, C * H * W, stride * W, stride)
    strides = x.itemsize * np.array(strides)
    x_stride = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
    x_cols = np.ascontiguousarray(x_stride)
    x_cols.shape = (C * HH * WW, N * out_h * out_w)

    # Теперь все свёртки сводятся к одному большому матричному умножению
    res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)

    # Приводим выход к нужной форме
    res.shape = (F, N, out_h, out_w)
    out = res.transpose(1, 0, 2, 3)

    # Для удобства возвращаем непрерывный в памяти (contiguous) массив
    # (старая версия conv_forward_fast этого не делала)
    out = np.ascontiguousarray(out)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache


def conv_backward_strides(dout, cache):
    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param["stride"], conv_param["pad"]

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, out_h, out_w = dout.shape

    db = np.sum(dout, axis=(0, 2, 3))

    dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)
    dx_cols.shape = (C, HH, WW, N, out_h, out_w)
    dx = col2im_6d(dx_cols, N, C, H, W, HH, WW, pad, stride)

    return dx, dw, db


def conv_backward_im2col(dout, cache):
    """
    Быстрая реализация обратного прохода для свёрточного слоя на основе
    im2col и col2im.
    """
    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param["stride"], conv_param["pad"]

    db = np.sum(dout, axis=(0, 2, 3))

    num_filters, _, filter_height, filter_width = w.shape
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
    dx = col2im_indices(dx_cols, x.shape, filter_height, filter_width, pad, stride)

    return dx, dw, db


conv_forward_fast = conv_forward_strides
conv_backward_fast = conv_backward_strides


def max_pool_forward_fast(x, pool_param):
    """
    Быстрая реализация прямого прохода для слоя max-пулинга.

    Функция выбирает между методом reshape и методом im2col. Если области
    пулинга квадратные и замощают входное изображение (покрывают его без
    перекрытий и пропусков), можно использовать очень быстрый метод reshape.
    Иначе используется запасной вариант — метод im2col, который ненамного
    быстрее наивной реализации.
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]

    same_size = pool_height == pool_width == stride
    tiles = H % pool_height == 0 and W % pool_width == 0
    if same_size and tiles:
        out, reshape_cache = max_pool_forward_reshape(x, pool_param)
        cache = ("reshape", reshape_cache)
    else:
        out, im2col_cache = max_pool_forward_im2col(x, pool_param)
        cache = ("im2col", im2col_cache)
    return out, cache


def max_pool_backward_fast(dout, cache):
    """
    Быстрая реализация обратного прохода для слоя max-пулинга.

    Функция выбирает метод reshape или метод im2col в зависимости от того,
    какой из них использовался при построении cache.
    """
    method, real_cache = cache
    if method == "reshape":
        return max_pool_backward_reshape(dout, real_cache)
    elif method == "im2col":
        return max_pool_backward_im2col(dout, real_cache)
    else:
        raise ValueError('Неизвестный метод "%s"' % method)


def max_pool_forward_reshape(x, pool_param):
    """
    Быстрая реализация прямого прохода для слоя max-пулинга, основанная на
    хитром изменении формы массива (reshape).

    Подходит только для квадратных областей пулинга, которые замощают вход.
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]
    assert pool_height == pool_width == stride, "Недопустимые параметры пулинга"
    assert H % pool_height == 0
    assert W % pool_width == 0
    x_reshaped = x.reshape(
        N, C, H // pool_height, pool_height, W // pool_width, pool_width
    )
    out = x_reshaped.max(axis=3).max(axis=4)

    cache = (x, x_reshaped, out)
    return out, cache


def max_pool_backward_reshape(dout, cache):
    """
    Быстрая реализация обратного прохода для слоя max-пулинга, основанная на
    хитром использовании транслирования (broadcasting) и изменения формы
    массива (reshape).

    Её можно использовать, только если прямой проход вычислялся с помощью
    max_pool_forward_reshape.

    ПРИМЕЧАНИЕ: если максимум в области пулинга достигается сразу в нескольких
    элементах входа, этот метод передаёт градиент ВСЕМ таким элементам, а не
    выбирает какой-то один. Если бы каждый из них получал весь градиент,
    пришедший сверху, градиент был бы неверным (впрочем, на практике такое
    случается редко). Поэтому градиент, пришедший сверху, делится поровну
    между всеми такими элементами — это даёт корректный субградиент. Деление
    выполняет строка dx_reshaped /= np.sum(mask, ...) ниже; оно заметно
    замедляет функцию (примерно на 40%).
    """
    x, x_reshaped, out = cache

    dx_reshaped = np.zeros_like(x_reshaped)
    out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
    mask = x_reshaped == out_newaxis
    dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
    dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
    dx_reshaped[mask] = dout_broadcast[mask]
    dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
    dx = dx_reshaped.reshape(x.shape)

    return dx


def max_pool_forward_im2col(x, pool_param):
    """
    Реализация прямого прохода max-пулинга на основе im2col.

    Она ненамного быстрее наивной версии, поэтому по возможности её лучше
    не использовать.
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]

    assert (H - pool_height) % stride == 0, "Недопустимая высота"
    assert (W - pool_width) % stride == 0, "Недопустимая ширина"

    out_height = (H - pool_height) // stride + 1
    out_width = (W - pool_width) // stride + 1

    x_split = x.reshape(N * C, 1, H, W)
    x_cols = im2col_indices(x_split, pool_height, pool_width, padding=0, stride=stride)
    x_cols_argmax = np.argmax(x_cols, axis=0)
    x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
    out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)

    cache = (x, x_cols, x_cols_argmax, pool_param)
    return out, cache


def max_pool_backward_im2col(dout, cache):
    """
    Реализация обратного прохода max-пулинга на основе im2col.

    Она ненамного быстрее наивной версии, поэтому по возможности её лучше
    не использовать.
    """
    x, x_cols, x_cols_argmax, pool_param = cache
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]

    dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
    dx_cols = np.zeros_like(x_cols)
    dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
    dx = col2im_indices(
        dx_cols, (N * C, 1, H, W), pool_height, pool_width, padding=0, stride=stride
    )
    dx = dx.reshape(x.shape)

    return dx
