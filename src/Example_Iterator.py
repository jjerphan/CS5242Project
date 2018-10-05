import numpy as np
import keras
import os

from discretization import load_nparray, make_cube, is_positive
from settings import resolution_cube, nb_channels, nb_examples, training_examples_folder, progress


class Example_Iterator(keras.utils.Sequence):
    """
        Generates data for Keras
    """

    def __init__(self, examples_folder, batch_size=32, dim=(resolution_cube, resolution_cube, resolution_cube),
                 n_channels=nb_channels,
                 n_classes=2, shuffle=True):

        self.dim = dim
        self.batch_size = batch_size
        self.examples_folder = examples_folder
        self.examples_files = sorted(os.listdir(examples_folder))
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch.

        :return:
        """
        return int(np.floor(len(self.examples_files) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data

        :param index:
        :return:
        """
        # Getting batch in the other
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        files_to_use = [self.examples_files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(files_to_use)

        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch

        :return:
        """
        self.indexes = np.arange(len(self.examples_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, files_to_use):
        """
        :param list_IDs_temp:
        :return:
        """
        """
        Return the first nb_examples cubes with their ys.
        :param nb_examples:
        :return: list of cubes and list of their ys
        """

        cubes = []
        ys = []
        for index, ex_file in enumerate(files_to_use):
            file_name = os.path.join(self.examples_folder, ex_file)
            example = load_nparray(file_name)

            cube = make_cube(example, resolution_cube)
            y = 1 * is_positive(ex_file)

            cubes.append(cube)
            ys.append(y)

        # Conversion to np.ndarrays with the first axes used for examples
        cubes = np.array(cubes)
        ys = np.array(ys)
        assert (ys.shape[0] == len(files_to_use))
        assert (cubes.shape[0] == len(files_to_use))

        return cubes, ys


if __name__ == "__main__":

    dataGenerator = Example_Iterator(training_examples_folder)

    for cubes, y in progress(dataGenerator):
        print(cubes.shape, np.mean(y))
