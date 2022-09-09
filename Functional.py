import numpy as np


def random_img_split(img):
    h, w = img.shape[0], img.shape[1]
    random_point = np.random.uniform(low=0, high=1)
    h_split, w_split = round(h * random_point), round(w * random_point)

    if h_split - 47 <= 0:
        h_split = h_split + abs(h_split-48)

    if w_split - 27 <= 0:
        w_split = w_split + abs(w_split-28)

    new_img = img[h_split-47:h_split, w_split-27:w_split]
    return new_img