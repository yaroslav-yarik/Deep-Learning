import os
import pickle
import tarfile
import urllib.request

import numpy as np

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "datasets", "cifar-10-batches-py"
)
CIFAR10_FILES = ["data_batch_%d" % i for i in range(1, 6)] + ["test_batch"]


def ensure_cifar10(root=CIFAR10_DIR):
    """
    Гарантирует, что набор данных CIFAR-10 есть в папке root.

    Набор данных поставляется вместе с лабораторной работой, поэтому обычно эта
    функция лишь проверяет, что все файлы на месте. Если каких-то файлов не
    хватает, набор данных скачивается (около 170 МБ) и распаковывается.

    Возвращает путь к папке с набором данных.
    """
    if all(os.path.isfile(os.path.join(root, f)) for f in CIFAR10_FILES):
        return root

    parent = os.path.dirname(os.path.abspath(root))
    os.makedirs(parent, exist_ok=True)
    archive = os.path.join(parent, "cifar-10-python.tar.gz")
    print("CIFAR-10 не найден в %s, скачиваем его (около 170 МБ)..." % root)
    urllib.request.urlretrieve(CIFAR10_URL, archive)
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(parent, filter="data")
        except TypeError:  # версия Python без фильтров распаковки
            tar.extractall(parent)
    os.remove(archive)
    extracted = os.path.join(parent, "cifar-10-batches-py")
    if os.path.abspath(extracted) != os.path.abspath(root):
        os.replace(extracted, root)
    print("Готово.")
    return root


def load_pickle(f):
    return pickle.load(f, encoding="latin1")


def load_CIFAR_batch(filename):
    """ Загружает один батч (файл) CIFAR-10 """
    with open(filename, "rb") as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ Загружает весь набор данных CIFAR-10 """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(
    num_training=49000, num_validation=1000, num_test=1000, subtract_mean=True
):
    """
    Загружает набор данных CIFAR-10 с диска и выполняет предобработку, чтобы
    подготовить его для классификаторов. Это те же шаги, что применялись для
    softmax-классификатора, только собранные в одну функцию.
    """
    # Загружаем исходные данные CIFAR-10
    cifar10_dir = ensure_cifar10()
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Формируем подвыборки
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Нормализуем данные: вычитаем среднее изображение
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Транспонируем, чтобы каналы шли первыми
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Упаковываем данные в словарь
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
