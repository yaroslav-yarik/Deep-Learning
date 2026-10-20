import hashlib
import os
import pickle
import tarfile
import urllib.request

import numpy as np

# Два источника одного и того же архива: сервер курса (быстрее из России) и
# официальный сайт. Выбор — переменная окружения DLCV_DATASET_SOURCE:
# "racoon", "official" или "auto" (по умолчанию: сначала сервер курса).
CIFAR10_SOURCES = {
    "racoon": "https://edu-s3.rracoon.ru/datasets/cifar-10-python.tar.gz",
    "official": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
}
# MD5 архива, опубликованный на https://www.cs.toronto.edu/~kriz/cifar.html
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "datasets", "cifar-10-batches-py"
)
CIFAR10_FILES = ["data_batch_%d" % i for i in range(1, 6)] + ["test_batch"]


def _download(url, path):
    """Скачивает url в path и проверяет MD5; при ошибке удаляет файл."""
    digest = hashlib.md5()
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(path, "wb") as out:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != CIFAR10_MD5:
            raise ValueError("контрольная сумма архива не совпала")
    except Exception:
        if os.path.exists(path):
            os.remove(path)
        raise


def ensure_cifar10(root=CIFAR10_DIR, source=None):
    """
    Гарантирует, что набор данных CIFAR-10 есть в папке root.

    Если каких-то файлов не хватает, архив (около 170 МБ) скачивается и
    распаковывается. source — "racoon" (сервер курса), "official" (сайт
    авторов) или "auto": сервер курса, а при ошибке — официальный сайт.
    По умолчанию берётся из переменной окружения DLCV_DATASET_SOURCE.

    Возвращает путь к папке с набором данных.
    """
    if all(os.path.isfile(os.path.join(root, f)) for f in CIFAR10_FILES):
        return root

    source = source or os.environ.get("DLCV_DATASET_SOURCE", "auto")
    if source == "auto":
        order = ["racoon", "official"]
    elif source in CIFAR10_SOURCES:
        order = [source]
    else:
        raise ValueError(
            "DLCV_DATASET_SOURCE должен быть racoon, official или auto, а не %r" % source
        )

    parent = os.path.dirname(os.path.abspath(root))
    os.makedirs(parent, exist_ok=True)
    archive = os.path.join(parent, "cifar-10-python.tar.gz")
    errors = []
    for name in order:
        url = CIFAR10_SOURCES[name]
        print("CIFAR-10 не найден, скачиваем (около 170 МБ): %s" % url)
        try:
            _download(url, archive)
            break
        except Exception as error:
            errors.append("%s: %s" % (url, error))
            print("Не удалось скачать с %s: %s" % (url, error))
    else:
        raise RuntimeError(
            "Не удалось скачать CIFAR-10:\n  " + "\n  ".join(errors)
            + "\nСкачайте архив вручную и распакуйте его в " + parent
        )
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
