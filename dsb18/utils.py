import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects

from .processing import binarize_mask


def predict(model, data, thres=0.5, min_size=20, **kwargs):
    keys = data.keys()

    x = np.array([data[key]["image"] for key in keys])
    x = np.expand_dims(x, axis=3)
    
    y = model.predict(x, **kwargs)
    y = np.squeeze(y)

    for key, mask in zip(keys, y):
        mask = binarize_mask(mask, thres)
        mask = binary_fill_holes(mask)
        mask = remove_small_objects(mask, min_size)
        data[key]["mask"] = mask

    return data

def rle_encode(mask, transpose=True, return_string=True):
    if transpose:
        mask = np.transpose(mask)
    rle = np.concatenate([[0], mask.ravel(), [0]])
    rle, = np.where(rle[1:] != rle[:-1])
    rle += 1
    rle[1::2] -= rle[::2]
    assert len(rle) > 0
    if return_string:
        return " ".join(rle.astype(str))
    return rle
