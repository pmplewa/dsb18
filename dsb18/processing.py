import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from skimage.util import invert
from tqdm import tqdm


def binarize_mask(mask, thres=0):
    return (mask > thres).astype(float)

def process_image(image, output_shape):
    image = rgb2gray(image)
    image = resize(image, output_shape)
    image = rescale_intensity(image)
    return image

def process_mask(mask, output_shape):
    mask = np.sum(mask, axis=0)
    mask = resize(mask, output_shape)
    mask = binarize_mask(mask)
    return mask

def process_data(input_data, output_shape=(256, 256)):
    data = {}
    
    for key in tqdm(input_data, desc="Processing data"):
        d = input_data[key]

        image = process_image(d["image"], output_shape)

        if "mask" in d:
            mask = process_mask(d["mask"], output_shape)

            # invert negative images
            positive = image[mask == 1]
            negative = image[mask == 0]
            if np.mean(negative) > np.mean(positive):
                image = invert(image)

            data[key] = {"image": image, "mask": mask}
        else:
            data[key] = {"image": image}
    
    return data
