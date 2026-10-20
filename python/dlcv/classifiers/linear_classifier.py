import os
import numpy as np
from .softmax import softmax_loss_vectorized


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        """
        Обучает линейный классификатор методом стохастического градиентного
        спуска.

        Входные данные:
        - X: массив numpy формы (N, D) с обучающими данными: N обучающих
          примеров, каждый размерности D.
        - y: массив numpy формы (N,) с обучающими метками; y[i] = c
          означает, что X[i] имеет метку 0 <= c < C, где C — число классов.
        - learning_rate: (float) скорость обучения (learning rate) для оптимизации.
        - reg: (float) коэффициент регуляризации.
        - num_iters: (integer) число шагов оптимизации
        - batch_size: (integer) число обучающих примеров на каждом шаге.
        - verbose: (boolean) если True, выводить ход оптимизации.

        Выходные данные:
        Список значений функции потерь на каждой итерации обучения.
        """
        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )  # предполагаем, что y принимает значения 0...K-1, где K — число классов
        if self.W is None:
            # ленивая инициализация W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Оптимизируем W стохастическим градиентным спуском
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None
            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices]
            y_batch = y[indices]

            #########################################################################
            # TODO:                                                                 #
            # Случайно выберите batch_size элементов из обучающих данных вместе     #
            # с соответствующими метками для этого шага градиентного спуска.        #
            # Сохраните данные в X_batch, а соответствующие метки — в y_batch;      #
            # после выборки X_batch должен иметь форму (batch_size, dim),           #
            # а y_batch — форму (batch_size,)                                       #
            #                                                                       #
            # Подсказка: для генерации индексов используйте np.random.choice.       #
            # Выборка с возвращением работает быстрее, чем без возвращения.         #
            #########################################################################


            # вычисляем значение функции потерь и градиент
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W -= learning_rate * grad

            # обновляем параметры
            #########################################################################
            # TODO:                                                                 #
            # Обновите веса, используя градиент и скорость обучения.                #
            #########################################################################


            if verbose and it % 100 == 0:
                print("итерация %d / %d: потери %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Предсказывает метки для объектов с помощью обученных весов этого
        линейного классификатора.

        Входные данные:
        - X: массив numpy формы (N, D) с данными: N примеров, каждый
          размерности D.

        Возвращает:
        - y_pred: предсказанные метки для данных из X. y_pred — одномерный
          массив длины N, каждый элемент которого — целое число, номер
          предсказанного класса.
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Реализуйте этот метод. Сохраните предсказанные метки в y_pred.          #
        ###########################################################################

        y_pred = np.argmax(X @ self.W, axis=1)
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Вычисляет функцию потерь и её производную.
        Переопределяется в подклассах.

        Входные данные:
        - X_batch: массив numpy формы (N, D) — мини-батч из N объектов,
          каждый размерности D.
        - y_batch: массив numpy формы (N,) с метками мини-батча.
        - reg: (float) коэффициент регуляризации.

        Возвращает кортеж из:
        - значения функции потерь (одно число типа float)
        - градиента по self.W — массива той же формы, что и W
        """
        pass

    def save(self, fname):
      """Сохраняет параметры модели."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = {"W": self.W}
      np.save(fpath, params)
      print(fname, "сохранён.")
    
    def load(self, fname):
      """Загружает параметры модели."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "не найден.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.W = params["W"]
        print(fname, "загружен.")
        return True


class Softmax(LinearClassifier):
    """ Подкласс с функцией потерь softmax + перекрёстная энтропия """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
