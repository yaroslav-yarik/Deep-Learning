import numpy as np
from random import randrange


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    Наивная реализация вычисления численного градиента функции f в точке x
    - f — функция, принимающая один аргумент
    - x — точка (массив numpy), в которой вычисляется градиент
    """

    fx = f(x)  # значение функции в исходной точке
    grad = np.zeros_like(x)
    # перебираем все индексы x
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:

        # вычисляем функцию в точке x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # увеличиваем на h
        fxph = f(x)  # вычисляем f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # вычисляем f(x - h)
        x[ix] = oldval  # восстанавливаем исходное значение

        # вычисляем частную производную по центральной разностной формуле
        grad[ix] = (fxph - fxmh) / (2 * h)  # наклон
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # переходим к следующей координате

    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Вычисляет численный градиент для функции, которая принимает массив numpy
    и возвращает массив numpy.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    Выбирает несколько случайных элементов x и вычисляет численный градиент
    только по этим координатам, выводя его сравнение с аналитическим.
    """

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h  # увеличиваем на h
        fxph = f(x)  # вычисляем f(x + h)
        x[ix] = oldval - h  # уменьшаем на h
        fxmh = f(x)  # вычисляем f(x - h)
        x[ix] = oldval  # восстанавливаем исходное значение

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (
            abs(grad_numerical) + abs(grad_analytic)
        )
        print(
            "численный: %f аналитический: %f, относительная ошибка: %e"
            % (grad_numerical, grad_analytic, rel_error)
        )
