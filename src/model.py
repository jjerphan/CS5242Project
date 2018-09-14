import os
import numpy as np
from keras import Sequential
from keras.layers import Conv3D, ZeroPadding3D
from keras.optimizers import Adam
from keras.losses import MSE

from discretization import make_cube
from settings import resolution_cube, examples_data, float_type, comment_delimiter

if __name__ == "__main__":
    file_name = os.path.join(examples_data, "0001_0001.csv")
    example = np.loadtxt(file_name, dtype=float_type, comments=comment_delimiter)
    cube = make_cube(example, resolution_cube)
    print(cube.shape)
    # plot_cube(cube)

    model = Sequential()
    model.add(ZeroPadding3D(input_shape=(resolution_cube, resolution_cube, resolution_cube, 2)))
    model.add(Conv3D(filters=32, kernel_size=3, strides=(1, 1, 1)))
    model.build()

    y = [1]
    model.compile(optimizer=Adam(), loss=MSE)
    model.fit(cube, y)
