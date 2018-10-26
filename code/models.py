from keras import Input, Model
from keras.layers import Dense, Flatten, Conv3D, Activation, MaxPooling3D, Dropout, BatchNormalization

from settings import LENGTH_CUBE_SIDE, NB_CHANNELS

# Configurations of the shape of data
input_shape = (LENGTH_CUBE_SIDE, LENGTH_CUBE_SIDE, LENGTH_CUBE_SIDE, NB_CHANNELS)
data_format = "channels_last"


def ProtNet():
    """
    Our really first proposition.

    A classic 3D Convolutional layers + dense layers architecture
    :return:
    """

    pool_size = (2, 2, 2)
    dropout_rate = 0.5

    inputs = Input(shape=input_shape)
    x = Conv3D(kernel_size=(5, 5, 5), activation="relu", filters=64)(inputs)
    x = MaxPooling3D(pool_size=pool_size)(x)

    x = Conv3D(kernel_size=(3, 3, 3), activation="relu", filters=128)(x)
    x = MaxPooling3D(pool_size=pool_size)(x)

    x = Conv3D(kernel_size=(3, 3, 3), activation="relu", filters=256)(x)
    x = Flatten()(x)

    x = Dense(1000, activation="relu")(x)
    x = Dropout(rate=dropout_rate)(x)

    x = Dense(500, activation="relu")(x)
    x = Dropout(rate=dropout_rate)(x)

    x = Dense(200, activation="relu")(x)
    x = Dropout(rate=dropout_rate)(x)

    x = Dense(1)(x)
    outputs = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name="ProtNet")
    return model


def ProtNet07():
    """
    A second version with higher dropout.

    :return:
    """

    pool_size = (2, 2, 2)
    dropout_rate = 0.7

    inputs = Input(shape=input_shape)
    x = Conv3D(kernel_size=(5, 5, 5), activation="relu", filters=64)(inputs)
    x = MaxPooling3D(pool_size=pool_size)(x)

    x = Conv3D(kernel_size=(3, 3, 3), activation="relu", filters=128)(x)
    x = MaxPooling3D(pool_size=pool_size)(x)

    x = Conv3D(kernel_size=(3, 3, 3), activation="relu", filters=256)(x)
    x = Flatten()(x)

    x = Dense(1000, activation="relu")(x)
    x = Dropout(rate=dropout_rate)(x)

    x = Dense(500, activation="relu")(x)
    x = Dropout(rate=dropout_rate)(x)

    x = Dense(200, activation="relu")(x)
    x = Dropout(rate=dropout_rate)(x)

    x = Dense(1)(x)
    outputs = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name="ProtNet07")
    return model


def ProtNetBN():
    """
    A variant with BatchNormalization

    :return:
    """

    pool_size = (2, 2, 2)

    inputs = Input(shape=input_shape)
    x = BatchNormalization()(inputs)

    x = Conv3D(kernel_size=(5, 5, 5), activation="relu", filters=64)(x)
    x = MaxPooling3D(pool_size=pool_size)(x)
    x = BatchNormalization()(x)

    x = Conv3D(kernel_size=(3, 3, 3), activation="relu", filters=128)(x)
    x = MaxPooling3D(pool_size=pool_size)(x)
    x = BatchNormalization()(x)

    x = Conv3D(kernel_size=(3, 3, 3), activation="relu", filters=256)(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)

    x = Dense(1000, activation="relu")(x)
    x = BatchNormalization()(x)

    x = Dense(500, activation="relu")(x)
    x = BatchNormalization()(x)

    x = Dense(200, activation="relu")(x)
    x = BatchNormalization()(x)

    x = Dense(1)(x)
    outputs = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name="ProtNetBN")
    return model


def SimplerProtNet07():
    """
    A simpler network (bottlenecked at the end) with dropout.

    :return:
    """

    pool_size = (2, 2, 2)
    dropout_rate = 0.7

    inputs = Input(shape=input_shape)
    x = Conv3D(kernel_size=(5, 5, 5), activation="relu", filters=64)(inputs)
    x = MaxPooling3D(pool_size=pool_size)(x)

    x = Conv3D(kernel_size=(3, 3, 3), activation="relu", filters=128)(x)
    x = MaxPooling3D(pool_size=pool_size)(x)

    x = Conv3D(kernel_size=(3, 3, 3), activation="relu", filters=256)(x)
    x = Flatten()(x)

    x = Dense(128, activation="relu")(x)
    x = Dropout(rate=dropout_rate)(x)

    x = Dense(128, activation="relu")(x)
    x = Dropout(rate=dropout_rate)(x)

    x = Dense(32, activation="relu")(x)
    x = Dropout(rate=dropout_rate)(x)

    x = Dense(1)(x)
    outputs = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name="SimplerProtNet07")
    return model


def SimplerProtNetBN():
    """
    A simpler network (bottlenecked at the end) with batchnormalisation.

    :return:
    """

    pool_size = (2, 2, 2)

    inputs = Input(shape=input_shape)
    x = BatchNormalization()(inputs)

    x = Conv3D(kernel_size=(5, 5, 5), activation="relu", filters=64)(x)
    x = MaxPooling3D(pool_size=pool_size)(x)
    x = BatchNormalization()(x)

    x = Conv3D(kernel_size=(3, 3, 3), activation="relu", filters=128)(x)
    x = MaxPooling3D(pool_size=pool_size)(x)
    x = BatchNormalization()(x)

    x = Conv3D(kernel_size=(3, 3, 3), activation="relu", filters=256)(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)

    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)

    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)

    x = Dense(32, activation="relu")(x)
    x = BatchNormalization()(x)

    x = Dense(1)(x)
    outputs = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name="SimplerProtNetBN")
    return model


def ProtVGGNet():
    """
    VGG inspired model : small (3,3) filter and periodic pooling.

    Some dropout used.

    :return:
    """

    pool_size = (2, 2, 2)
    kernel_size = (3, 3, 3)
    dropout_value = 0.5

    inputs = Input(shape=input_shape)

    x = Conv3D(kernel_size=kernel_size, padding="same", activation="relu", filters=64)(inputs)
    x = Conv3D(kernel_size=kernel_size, padding="same", activation="relu", filters=64)(x)
    x = MaxPooling3D(pool_size=pool_size)(x)

    x = Conv3D(kernel_size=kernel_size, padding="same", activation="relu", filters=128)(x)
    x = Conv3D(kernel_size=kernel_size, padding="same", activation="relu", filters=128)(x)
    x = MaxPooling3D(pool_size=pool_size)(x)

    x = Conv3D(kernel_size=kernel_size, padding="same", activation="relu", filters=128)(x)
    x = Conv3D(kernel_size=kernel_size, padding="same", activation="relu", filters=128)(x)
    x = MaxPooling3D(pool_size=pool_size)(x)

    x = Conv3D(kernel_size=kernel_size, padding="same", activation="relu", filters=256)(x)
    x = Conv3D(kernel_size=kernel_size, padding="same", activation="relu", filters=256)(x)
    x = MaxPooling3D(pool_size=pool_size)(x)

    x = Flatten()(x)

    x = Dense(128, activation="relu")(x)
    x = Dropout(dropout_value)(x)

    x = Dense(48, activation="relu")(x)
    x = Dropout(dropout_value)(x)

    x = Dense(1)(x)
    outputs = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name="ProtVGGNet")
    return model


models_available = [ProtNet(), ProtNet07(), ProtNetBN(), SimplerProtNet07(), SimplerProtNetBN(), ProtVGGNet()]
models_available_names = list(map(lambda model: model.name, models_available))

if __name__ == "__main__":
    print(f"{len(models_available_names)} Models availables: \n\n")
    for i, model in enumerate(models_available):
        print(f"#{i}: {model.name}")
        model.summary()
