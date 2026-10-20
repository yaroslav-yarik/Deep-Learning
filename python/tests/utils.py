"""Общие помощники тестов (как в ноутбуках)."""
import numpy as np

from dlcv.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array

__all__ = ["rel_error", "eval_numerical_gradient", "eval_numerical_gradient_array"]


def rel_error(x, y):
    """Относительная ошибка, как в ноутбуках."""
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
