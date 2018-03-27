from keras import backend as K
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.core import Dropout
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.models import Input, Model


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return (2*K.sum(y_true*y_pred))/(K.sum(y_true)+K.sum(y_pred))

def unet(min_dim=64, input_shape=(256, 256), upsampling=False):
    inputs = Input(shape=input_shape+(1,))
    
    c1 = Conv2D(min_dim, (3, 3), activation="relu", padding="same")(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(min_dim, (3, 3), activation="relu", padding="same")(c1)
    p1 = MaxPooling2D()(c1)
    
    c2 = Conv2D(2*min_dim, (3, 3), activation="relu", padding="same")(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(2*min_dim, (3, 3), activation="relu", padding="same")(c2)
    p2 = MaxPooling2D()(c2)
    
    c3 = Conv2D(4*min_dim, (3, 3), activation="relu", padding="same")(p2)
    c3 = Dropout(0.1)(c3)
    c3 = Conv2D(4*min_dim, (3, 3), activation="relu", padding="same")(c3)
    p3 = MaxPooling2D()(c3)
    
    c4 = Conv2D(8*min_dim, (3, 3), activation="relu", padding="same")(p3)
    c4 = Dropout(0.1)(c4)
    c4 = Conv2D(8*min_dim, (3, 3), activation="relu", padding="same")(c4)
    p4 = MaxPooling2D()(c4)
    
    c5 = Conv2D(16*min_dim, (3, 3), activation="relu", padding="same")(p4)
    c5 = Dropout(0.1)(c5)
    c5 = Conv2D(16*min_dim, (3, 3), activation="relu", padding="same")(c5)
    
    if upsampling:
      u6 = UpSampling2D((2, 2))(c5)
      u6 = Conv2D(8*min_dim, (2, 2), activation="relu", padding="same")(u6)
    else:
      u6 = Conv2DTranspose(8*min_dim, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(8*min_dim, (3, 3), activation="relu", padding="same")(u6)
    c6 = Dropout(0.1)(c6)
    c6 = Conv2D(8*min_dim, (3, 3), activation="relu", padding="same")(c6)
    
    if upsampling:
      u7 = UpSampling2D((2, 2))(c6)
      u7 = Conv2D(4*min_dim, (2, 2), activation="relu", padding="same")(u7)
    else:
      u7 = Conv2DTranspose(4*min_dim, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(4*min_dim, (3, 3), activation="relu", padding="same")(u7)
    c7 = Dropout(0.1)(c7)
    c7 = Conv2D(4*min_dim, (3, 3), activation="relu", padding="same")(c7)
    
    if upsampling:
      u8 = UpSampling2D((2, 2))(c7)
      u8 = Conv2D(2*min_dim, (2, 2), activation="relu", padding="same")(u8)
    else:
      u8 = Conv2DTranspose(2*min_dim, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(2*min_dim, (3, 3), activation="relu", padding="same")(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(2*min_dim, (3, 3), activation="relu", padding="same")(c8)
    
    if upsampling:
      u9 = UpSampling2D((2, 2))(c8)
      u9 = Conv2D(min_dim, (2, 2), activation="relu", padding="same")(u9)
    else:
      u9 = Conv2DTranspose(min_dim, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(min_dim, (3, 3), activation="relu", padding="same")(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(min_dim, (3, 3), activation="relu", padding="same")(c9)
    
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=[dice_coef, "accuracy", "mean_squared_error"])

    return model
