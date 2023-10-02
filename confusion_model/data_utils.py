import cv2
from PIL.Image import Image
import PIL.Image as img
import numpy as np
from typing import Tuple, List
from time import time


def convert_from_cv2_to_image(image: np.ndarray) -> Image:
    return img.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(
    image: Image, new_area: Tuple[int, int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    col_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
    if new_area is not None:
        col_img = cv2.resize(col_img, dsize=new_area, interpolation=cv2.INTER_LINEAR)
        gray_img = cv2.resize(gray_img, dsize=new_area, interpolation=cv2.INTER_LINEAR)
    return col_img, gray_img


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2 - t1):.4f}s")
        return result

    return wrap_func


def get_closest_ix(coordinate: float, coordinate_list: List[float]) -> int:
    """
    Given a coordinate associated with a face object, find the index of the closest
    face in a list of other coordinates
    :param coordinate: coordinate (any coordinate) of faces
    :param coordinate_list: List of all coordinates of faces
    :return: index of closest face to given face coordinate
    """
    return np.argmin(
        [abs(coordinate - new_coordinate) for new_coordinate in coordinate_list]
    )
