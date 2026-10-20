import os
import pickle

import numpy as np

from . import optim


class Solver(object):
    """
    Класс Solver инкапсулирует всю логику, необходимую для обучения моделей
    классификации. Solver выполняет стохастический градиентный спуск, используя
    различные правила обновления, определённые в optim.py.

    Solver принимает и обучающие, и валидационные данные с метками, поэтому
    может периодически проверять точность классификации на обеих выборках и
    вовремя заметить переобучение.

    Чтобы обучить модель, сначала создайте экземпляр Solver, передав в
    конструктор модель, набор данных и различные настройки (скорость обучения,
    размер батча и т. д.). Затем вызовите метод train(), чтобы запустить
    процедуру оптимизации и обучить модель.

    После завершения метода train() в model.params будут находиться параметры,
    показавшие лучший результат на валидационной выборке за всё время обучения.
    Кроме того, атрибут solver.loss_history будет содержать список всех
    значений потерь, полученных во время обучения, а атрибуты
    solver.train_acc_history и solver.val_acc_history — списки значений
    точности модели на обучающей и валидационной выборках после каждой эпохи.

    Пример использования может выглядеть примерно так:

    data = {
      'X_train': # обучающие данные
      'y_train': # метки обучающих данных
      'X_val': # валидационные данные
      'y_val': # метки валидационных данных
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-4,
                    },
                    lr_decay=0.95,
                    num_epochs=5, batch_size=200,
                    print_every=100)
    solver.train()


    Solver работает с объектом модели, который должен поддерживать следующий
    интерфейс (API):

    - model.params должен быть словарём, который сопоставляет строковым именам
      параметров массивы numpy со значениями этих параметров.

    - model.loss(X, y) должен быть функцией, которая на этапе обучения
      вычисляет значение потерь и градиенты, а на этапе тестирования — оценки
      (scores) классов. Входы и выходы у неё следующие:

      Входные данные:
      - X: массив формы (N, d_1, ..., d_k) — мини-батч входных данных
      - y: массив меток формы (N,), где y[i] — метка для X[i].

      Возвращает:
      Если y равен None, выполняет прямой проход в режиме тестирования и
      возвращает:
      - scores: массив формы (N, C) с оценками классов для X, где
        scores[i, c] — оценка класса c для X[i].

      Если y не равен None, выполняет прямой и обратный проходы в режиме
      обучения и возвращает кортеж из:
      - loss: скаляр — значение функции потерь
      - grads: словарь с теми же ключами, что и self.params, который
        сопоставляет именам параметров градиенты функции потерь по этим
        параметрам.
    """

    def __init__(self, model, data, **kwargs):
        """
        Создаёт новый экземпляр Solver.

        Обязательные аргументы:
        - model: объект модели, поддерживающий описанный выше интерфейс
        - data: словарь с обучающими и валидационными данными, содержащий:
          'X_train': массив формы (N_train, d_1, ..., d_k) с обучающими изображениями
          'X_val': массив формы (N_val, d_1, ..., d_k) с валидационными изображениями
          'y_train': массив формы (N_train,) с метками обучающих изображений
          'y_val': массив формы (N_val,) с метками валидационных изображений

        Необязательные аргументы:
        - update_rule: строка с именем правила обновления из optim.py.
          По умолчанию 'sgd'.
        - optim_config: словарь с гиперпараметрами, которые будут переданы
          выбранному правилу обновления. Разным правилам обновления нужны
          разные гиперпараметры (см. optim.py), но всем им нужен параметр
          'learning_rate', поэтому он должен присутствовать всегда.
        - lr_decay: скаляр для затухания скорости обучения; после каждой эпохи
          скорость обучения умножается на это значение.
        - batch_size: размер мини-батчей, на которых во время обучения
          вычисляются потери и градиент.
        - num_epochs: число эпох обучения.
        - print_every: целое число; значение потерь на обучении выводится
          каждые print_every итераций.
        - verbose: логическое значение; если False, то во время обучения
          ничего не выводится.
        - num_train_samples: число обучающих примеров, по которым проверяется
          точность на обучающей выборке; по умолчанию 1000; None — использовать
          всю обучающую выборку.
        - num_val_samples: число валидационных примеров, по которым проверяется
          точность на валидации; по умолчанию None — используется вся
          валидационная выборка.
        - checkpoint_name: если не None, то после каждой эпохи чекпоинт модели
          сохраняется в файл с этим префиксом имени.
        """
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]

        # Разбираем именованные аргументы
        self.update_rule = kwargs.pop("update_rule", "sgd")
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.num_train_samples = kwargs.pop("num_train_samples", 1000)
        self.num_val_samples = kwargs.pop("num_val_samples", None)

        self.checkpoint_name = kwargs.pop("checkpoint_name", None)
        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)

        # Если переданы лишние именованные аргументы, выбрасываем ошибку
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Неизвестные аргументы %s" % extra)

        # Проверяем, что такое правило обновления существует, и заменяем
        # строковое имя на саму функцию
        if not hasattr(optim, self.update_rule):
            raise ValueError('Неизвестное правило обновления update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        """
        Инициализирует служебные переменные для оптимизации. Не вызывайте этот
        метод вручную.
        """
        # Служебные переменные для учёта хода обучения
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Создаём для каждого параметра собственную копию optim_config
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        Выполняет один шаг обновления параметров по градиенту. Этот метод
        вызывается из train(), вызывать его вручную не нужно.
        """
        # Формируем мини-батч из обучающих данных
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Вычисляем значение потерь и градиент
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Обновляем параметры
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        checkpoint = {
            "model": self.model,
            "update_rule": self.update_rule,
            "lr_decay": self.lr_decay,
            "optim_config": self.optim_config,
            "batch_size": self.batch_size,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "epoch": self.epoch,
            "loss_history": self.loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history,
        }
        filename = "%s_epoch_%d.pkl" % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Сохраняем чекпоинт в "%s"' % filename)
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Проверяет точность модели на переданных данных.

        Входные данные:
        - X: массив данных формы (N, d_1, ..., d_k)
        - y: массив меток формы (N,)
        - num_samples: если не None, то берётся случайная подвыборка данных и
          модель проверяется только на num_samples объектах.
        - batch_size: X и y разбиваются на батчи такого размера, чтобы не
          расходовать слишком много памяти.

        Возвращает:
        - acc: скаляр — доля объектов, которые модель классифицировала
          правильно.
        """

        # При необходимости берём подвыборку данных
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Вычисляем предсказания по батчам
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        """
        Запускает оптимизацию, чтобы обучить модель.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Время от времени выводим значение потерь на обучении
            if self.verbose and t % self.print_every == 0:
                print(
                    "(Итерация %d / %d) потери: %f"
                    % (t + 1, num_iterations, self.loss_history[-1])
                )

            # В конце каждой эпохи увеличиваем счётчик эпох и применяем
            # затухание скорости обучения.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay

            # Проверяем точность на обучающей и валидационной выборках на
            # первой итерации, на последней итерации и в конце каждой эпохи.
            first_it = t == 0
            last_it = t == num_iterations - 1
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(
                    self.X_train, self.y_train, num_samples=self.num_train_samples
                )
                val_acc = self.check_accuracy(
                    self.X_val, self.y_val, num_samples=self.num_val_samples
                )
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()

                if self.verbose:
                    print(
                        "(Эпоха %d / %d) точность на обучении: %f; на валидации: %f"
                        % (self.epoch, self.num_epochs, train_acc, val_acc)
                    )

                # Запоминаем лучшую модель
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # В конце обучения записываем в модель лучшие найденные параметры
        self.model.params = self.best_params
