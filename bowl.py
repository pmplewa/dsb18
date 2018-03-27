from argparse import ArgumentParser
from pathlib import Path

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import numpy as np

from dsb18.augmentation import data_gen
from dsb18.io import import_data, output_table, output_images
from dsb18.models.unet import unet, dice_coef
#from dsb18.plotting import plot_example
from dsb18.processing import process_data
from dsb18.utils import predict


def run(data_dir, output_dir, mode, checkpoint, epochs, batch_size):
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    assert data_path.is_dir(), "Input directory does not exist"
    assert output_path.is_dir(), "Ouput directory does not exist"
    
    if mode == "train":
        assert epochs is not None, "'epochs' is a required parameter"

        # data import & processing
        train_data = import_data(data_path.joinpath("stage1_train"))
        train_data = process_data(train_data)
    
        # data augmentation
        train_batch = data_gen(train_data, seed=42, batch_size=batch_size)
        
        # model setup
        model = unet()
        model.summary()
    
        # model fitting
        checkpoint_path = output_path.joinpath("checkpoint.{epoch:02d}.h5")
        fit = model.fit_generator(
            generator=train_batch,
            epochs=epochs,                          
            steps_per_epoch=int(np.ceil(len(train_data)/batch_size)),
            verbose=1,
            callbacks=[
                ModelCheckpoint(checkpoint_path, period=1, verbose=1)])
    
    elif mode == "test":
        assert checkpoint is not None, "'checkpoint' is a required parameter"

        # data import & processing
        test_data = import_data(data_path.joinpath("stage1_test"))
        test_data = process_data(test_data)

        # model setup
        checkpoint_path = Path(checkpoint)
        assert checkpoint_path.is_file(), "Could not find checkpoint"
        model = load_model(checkpoint_path,
            custom_objects={"dice_coef": dice_coef})

        # prediction
        pred_data = predict(model, test_data, batch_size=batch_size, verbose=1)

        # data output
        output_table(output_path.joinpath("test_results.csv"), pred_data)
        output_images(output_path, pred_data)

    else:
      raise Exception("Mode must be 'train' or 'test'")

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--mode", required=True, choices=["train", "test"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    opts = parse_arguments()
    run(**vars(opts))
