import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter


def extract_features(imgs, feature_fns, verbose=False):
    """
    По пиксельным данным изображений и нескольким функциям признаков, каждая из
    которых обрабатывает одно изображение, применяет все функции признаков ко
    всем изображениям, конкатенирует векторы признаков каждого изображения и
    записывает признаки всех изображений в одну матрицу.

    Входные данные:
    - imgs: массив N x H x W x C с пиксельными данными N изображений.
    - feature_fns: список из k функций признаков. i-я функция признаков должна
      принимать на вход массив H x W x D и возвращать (одномерный) массив
      длины F_i.
    - verbose: логическое значение; если True, выводить ход выполнения.

    Возвращает:
    Массив формы (N, F_1 + ... + F_k), каждая строка которого — конкатенация
    всех признаков одного изображения.
    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # По первому изображению определяем размерности признаков
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, "Функции признаков должны возвращать одномерные массивы"
        feature_dims.append(feats.size)
        first_image_features.append(feats)

    # Теперь, когда размерности признаков известны, можно выделить один
    # большой массив, в строках которого будут храниться признаки изображений.
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.hstack(first_image_features).T

    # Извлекаем признаки для остальных изображений.
    for i in range(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim
            imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        if verbose and i % 1000 == 999:
            print("Признаки извлечены для %d / %d изображений" % (i + 1, num_images))

    return imgs_features


def rgb2gray(rgb):
    """Переводит RGB-изображение в оттенки серого

      Параметры:
        rgb : RGB-изображение

      Возвращает:
        gray : изображение в оттенках серого

    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def hog_feature(im):
    """Вычисляет для изображения признаки HOG (гистограмма направленных градиентов)

         Изменённая версия skimage.feature.hog
         http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog

       Литература:
         Histograms of Oriented Gradients for Human Detection
         Navneet Dalal and Bill Triggs, CVPR 2005

      Параметры:
        im : входное изображение в оттенках серого или RGB

      Возвращает:
        feat: признаки HOG (гистограмма направленных градиентов)

    """

    # при необходимости переводим RGB в оттенки серого
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.atleast_2d(im)

    sx, sy = image.shape  # размер изображения
    orientations = 9  # число бинов по направлению градиента
    cx, cy = (8, 8)  # число пикселей в ячейке

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # градиент по направлению x
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # градиент по направлению y
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # модуль градиента
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # направление градиента

    n_cellsx = int(np.floor(sx / cx))  # число ячеек по x
    n_cellsy = int(np.floor(sy / cy))  # число ячеек по y
    # вычисляем интегральные изображения для каждого направления
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        # создаём новое интегральное изображение для этого направления:
        # оставляем только направления из текущего диапазона
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1), grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i, temp_ori, 0)
        # выбираем модули градиента для этих направлений
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        orientation_histogram[:, :, i] = uniform_filter(temp_mag, size=(cx, cy))[
            round(cx / 2) :: cx, round(cy / 2) :: cy
        ].T

    return orientation_histogram.ravel()


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """
    Вычисляет цветовую гистограмму изображения по тону (hue).

    Входные данные:
    - im: массив H x W x C с пиксельными данными RGB-изображения.
    - nbin: число бинов гистограммы. (по умолчанию: 10)
    - xmin: минимальное значение пикселя (по умолчанию: 0)
    - xmax: максимальное значение пикселя (по умолчанию: 255)
    - normalized: нормировать ли гистограмму (по умолчанию: True)

    Возвращает:
      Одномерный вектор длины nbin — цветовую гистограмму по тону (hue)
      входного изображения.
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    hsv = matplotlib.colors.rgb_to_hsv(im / xmax) * xmax
    imhist, bin_edges = np.histogram(hsv[:, :, 0], bins=bins, density=normalized)
    imhist = imhist * np.diff(bin_edges)

    # возвращаем гистограмму
    return imhist
