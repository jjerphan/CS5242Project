from keras import Sequential
from keras.layers import Dense, Flatten, Conv3D
import keras.backend as K
from keras.losses import MSE

from settings import resolution_cube, nb_features


def first_model():
    # Configurations of the shape of data

    input_shape = (resolution_cube, resolution_cube, resolution_cube, nb_features-3)
    data_format = "channels_last"

    # Defining the model
    model = Sequential()
    # model.add(ZeroPadding3D()) # TODO : add this layer
    model.add(Conv3D(
        kernel_size=3,
        input_shape=input_shape,
        filters=32,
        data_format=data_format
    ))
    model.add(Flatten())
    model.add(Dense(3 * resolution_cube))
    model.add(Dense(2 * resolution_cube))
    model.add(Dense(1 * resolution_cube))
    model.add(Dense(1, activation='sigmoid'))
    model.build()

    def mean_pred(y_true, y_pred):
        return K.mean(y_pred)

    model.compile(optimizer='rmsprop', loss=MSE, metrics=['accuracy', mean_pred])

    return model
