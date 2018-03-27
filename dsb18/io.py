from pathlib import Path

import numpy as np
import pandas as pd
from skimage.io import imread, imread_collection, imsave
from tqdm import tqdm

from .utils import rle_encode


def import_data(data_path):
    data = {}
    
    for path in tqdm(data_path.iterdir(), desc="Loading data"):
        if path.is_dir():
            d = {}
        
            image_dir = path.joinpath("images")
            assert image_dir.is_dir()
            image_path = list(image_dir.glob("*"))
            assert len(image_path) == 1
            d["image"] = imread(image_path[0])    
                
            mask_dir = path.joinpath("masks")
            if mask_dir.is_dir():
                mask_path = list(mask_dir.glob("*"))
                assert len(mask_path) > 0
                d["mask"] = imread_collection(mask_path).concatenate()
                
            data[path.name] = d
                
    return data

def output_table(path, data):
    df = pd.DataFrame([{
        "ImageId": key,
        "EncodedPixels": rle_encode(data[key]["mask"])}
        for key in data])

    df[["ImageId", "EncodedPixels"]].to_csv(path, index=False)
    
    return df

def output_images(path, data):
    for key in tqdm(data, desc="Saving images"):
        image = data[key]["image"]
        mask = data[key]["mask"]
        imsave(path.joinpath(f"{key}.png"), np.hstack((image, mask)))
