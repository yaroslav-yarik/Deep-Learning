"""
Вспомогательные функции для загрузки изображений.
"""
import io
import urllib.error
import urllib.request

import numpy as np
from PIL import Image


def image_from_url(url):
    """
    Загружает изображение по URL. Возвращает массив numpy со значениями
    пикселей или None, если изображение не удалось скачать.
    """
    try:
        with urllib.request.urlopen(url, timeout=15) as f:
            data = f.read()
        return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    except urllib.error.HTTPError as e:
        print("Ошибка HTTP: ", e.code, url)
    except urllib.error.URLError as e:
        print("Ошибка URL: ", e.reason, url)
    except (OSError, ValueError) as e:
        print("Ошибка: ", e, url)
    return None
