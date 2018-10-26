import keras
from keras import Input, Model
from keras.layers import Dense, Flatten, Conv3D, Activation, MaxPooling3D, Dropout, BatchNormalization, \
    AveragePooling2D, AveragePooling3D
from keras.regularizers import l2

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


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 use_batch_norm=True):
    """
    Return a ResNet layer using the given parameters

    :param inputs:
    :param num_filters:
    :param kernel_size:
    :param strides:
    :param activation:
    :param use_batch_norm:
    :return:
    """

    conv = Conv3D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    x = conv(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x


def ProtResNet():
    """
    ResNet inspired model.

    Modified implementation : https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py#L116

    :return:
    """
    depth = 22 # can be 20, 32, 44

    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 use_batch_norm=False)
            # Shortcut connection here
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    x = AveragePooling3D(pool_size=5)(x)
    y = Flatten()(x)
    outputs = Dense(1,
                    activation='sigmoid',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs,name="ProtResNet")
    return model


def ProtInceptionNet():
    """
    Inception inspired model.


    :return:
    """
    dropout_value = 0.5
    nb_inception_modules = 3
    nb_filters = 16

    inputs = Input(shape=input_shape)

    x = Conv3D(nb_filters, (1, 1, 1), padding='same', activation='relu')(inputs)
    x = Conv3D(nb_filters, (1, 1, 1), padding='same', activation='relu')(x)
    x = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)

    for num_module in range(1, nb_inception_modules+1):

        sub_mod_1 = Conv3D(nb_filters*num_module, (1, 1, 1), padding='same', activation='relu')(x)
        sub_mod_1 = Conv3D(nb_filters*num_module, (3, 3, 3), padding='same', activation='relu')(sub_mod_1)

        sub_mod_2 = Conv3D(nb_filters*num_module, (1, 1, 1), padding='same', activation='relu')(x)
        sub_mod_2 = Conv3D(nb_filters*num_module, (5, 5, 5), padding='same', activation='relu')(sub_mod_2)

        sub_mod_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        sub_mod_3 = Conv3D(nb_filters*num_module, (1, 1, 1), padding='same', activation='relu')(sub_mod_3)

        y = keras.layers.concatenate([sub_mod_1, sub_mod_2, sub_mod_3], axis=4)

        # To reduce dimensions:
        x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(y)

    x = Flatten()(x)

    x = Dense(128, activation="relu")(x)
    x = Dropout(dropout_value)(x)

    x = Dense(48, activation="relu")(x)
    x = Dropout(dropout_value)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs,name="ProtInceptionNet")

    return model


models_available = [ProtNet(), ProtNet07(), ProtNetBN(),
                    SimplerProtNet07(), SimplerProtNetBN(),
                    ProtVGGNet(), ProtResNet(), ProtInceptionNet()]
models_available_names = list(map(lambda model: model.name, models_available))

if __name__ == "__main__":
    print(f"{len(models_available_names)} Models availables: \n\n")
    for i, model in enumerate(models_available):
        print(f"#{i}: {model.name}")
        model.summary()
