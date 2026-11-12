import numpy as np

from . import optim
from .coco_utils import sample_coco_minibatch, decode_captions

import torch


class CaptioningSolverPytorch(object):
    """
    Класс CaptioningSolverPytorch инкапсулирует всю логику, необходимую для
    обучения моделей генерации подписей к изображениям, написанных на PyTorch.

    Чтобы обучить модель, сначала создайте экземпляр CaptioningSolverPytorch,
    передав в конструктор модель, набор данных и различные настройки (скорость
    обучения, размер батча и т. д.). Затем вызовите метод train(), чтобы
    запустить процедуру оптимизации и обучить модель.

    После завершения метода train() атрибут solver.loss_history будет
    содержать список всех значений потерь, полученных во время обучения.

    Пример использования может выглядеть примерно так:

    data = load_coco_data()
    model = MyAwesomeModel(hidden_dim=100)
    solver = CaptioningSolverPytorch(model, data,
                    learning_rate=1e-3,
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()


    CaptioningSolverPytorch работает с объектом модели, который должен
    поддерживать следующий интерфейс (API):

    - model.params должен быть словарём, который сопоставляет строковым именам
      параметров тензоры PyTorch со значениями этих параметров.

    - model.loss(features, captions) должен быть функцией, которая вычисляет
      значение потерь на мини-батче. Входы и выходы у неё следующие:

      Входные данные:
      - features: массив формы (N, D) с признаками мини-батча изображений
      - captions: массив формы (N, T) с подписями к этим изображениям, где
        каждый элемент лежит в диапазоне [0, V).

      Возвращает:
      - loss: скалярный тензор PyTorch — значение функции потерь. Градиенты
        по параметрам вычисляет autograd при вызове loss.backward(), поэтому
        возвращать их не нужно.
    """

    def __init__(self, model, data, **kwargs):
        """
        Создаёт новый экземпляр CaptioningSolverPytorch.

        Обязательные аргументы:
        - model: объект модели, поддерживающий описанный выше интерфейс
        - data: словарь с обучающими и валидационными данными, который
          возвращает load_coco_data

        Необязательные аргументы:

        - learning_rate: скорость обучения для оптимизатора (Adam).
        - batch_size: размер мини-батчей, на которых во время обучения
          вычисляются потери и градиент.
        - num_epochs: число эпох обучения.
        - print_every: целое число; значение потерь на обучении выводится
          каждые print_every итераций.
        - verbose: логическое значение; если False, то во время обучения
          ничего не выводится.
        """
        self.model = model
        self.data = data

        # Разбираем именованные аргументы
        self.learning_rate = kwargs.pop("learning_rate", 0.001)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)

        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)
        self.optim = torch.optim.Adam(list(model.params.values()), self.learning_rate)

        # Если переданы лишние именованные аргументы, выбрасываем ошибку
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Неизвестные аргументы %s" % extra)

        self._reset()

    def _reset(self):
        """
        Инициализирует служебные переменные для оптимизации. Не вызывайте этот
        метод вручную.
        """
        # Служебные переменные для учёта хода обучения
        self.epoch = 0
        self.loss_history = []


    def _step(self):
        """
        Выполняет один шаг обновления параметров по градиенту. Этот метод
        вызывается из train(), вызывать его вручную не нужно.
        """
        # Формируем мини-батч из обучающих данных
        minibatch = sample_coco_minibatch(
            self.data, batch_size=self.batch_size, split="train"
        )
        captions, features, urls = minibatch

        captions = torch.from_numpy(captions).long()
        features = torch.from_numpy(features)
        loss = self.model.loss(features, captions)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.loss_history.append(loss.detach().numpy())

    def train(self):
        """
        Запускает оптимизацию, чтобы обучить модель.
        """
        for k, v in self.model.params.items():
          v.requires_grad_()

        num_train = self.data["train_captions"].shape[0]
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

            # В конце эпохи увеличиваем счётчик эпох.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1

        for k, v in self.model.params.items():
          v.requires_grad_(False)
