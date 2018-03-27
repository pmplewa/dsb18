from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.util import invert

from .processing import binarize_mask


def data_gen(data, seed, batch_size):
    keys = data.keys()

    x = np.array([data[key]["image"] for key in keys])
    x = np.expand_dims(x, axis=3)

    y = np.array([data[key]["mask"] for key in keys])
    y = np.expand_dims(y, axis=3)

    # parameters for geometric transfomration
    geom_args = {
        "rotation_range": 180,
        "width_shift_range": 1,
        "height_shift_range": 1,
        "shear_range": 0,
        "zoom_range": 0.1,
        "fill_mode": "reflect",
        "horizontal_flip": True,
        "vertical_flip": True}
    
    # parameters for intensity transformation
    int_args = {
        "saturation_range": None,
        "invert": True}
    
    image_gen = ImageDataGenerator(**geom_args)\
        .flow(x, seed=seed, batch_size=batch_size)
    mask_gen = ImageDataGenerator(**geom_args)\
        .flow(y, seed=seed, batch_size=batch_size)
    
    while True:
        # apply random distortion
        image, mask = next(zip(image_gen, mask_gen))
        
        assert len(image) == len(mask)
        for i in range(len(image)):
            image[i] = rescale_intensity(image[i])
            mask[i] = binarize_mask(mask[i])
        
            # apply random saturation threshold
            # (assumes input images are positive)
            if int_args["saturation_range"] is not None:
                clip_val = np.random.uniform(int_args["saturation_range"], 1)
                image[i] = np.clip(image[i], 0, clip_val)
                image[i] = rescale_intensity(image[i])
                
            # apply random inversion
            if int_args["invert"]:
                if np.random.randint(2):
                    image[i] = invert(image[i])
                 
        yield image, mask
