import os
from collections import defaultdict

import keras
import numpy as np

from discretization import load_nparray, make_cube, plot_cube
from pipeline_fixtures import is_positive, is_negative
from settings import max_nb_neg_per_pos, length_cube_side, training_examples_folder, shape_cube, validation_examples_folder

from mpl_toolkits.mplot3d import Axes3D


class ExamplesIterator(keras.utils.Sequence):
    """
    A class that loads data incrementally to feed Keras Model.

    The data is loaded by batch of size `batch_size`.

    We can control the amount of negative examples we can train the network on to
    make sure our network gets trained accordingly to a good distribution : need to be tweak though.

    We can choose to shuffle after each epoch or not too.

    """

    def __init__(self,
                 examples_folder: object,
                 batch_size: object = 32,
                 nb_neg: object = None,
                 shuffle_after_completion: object = True,
                 max_examples: object = None) -> object:
        """

        :param examples_folder: the folder containing the examples
        :param batch_size: the batch size to use
        :param nb_neg: the number of positive example (this controls the number of examples
        :param shuffle_after_completion: to shuffle the data or not after each epoch
        :param max_examples: if specified, just use the number of examples given
        """

        self.batch_size = batch_size
        self.examples_folder = examples_folder
        self.shuffle_after_completion = shuffle_after_completion

        all_files = sorted(os.listdir(examples_folder))
        pos_files = list(filter(is_positive, all_files))
        neg_files = list(filter(is_negative, all_files))

        nb_neg_files_per_pos_file = int(len(neg_files) / len(pos_files))

        if nb_neg is None:
            nb_neg = nb_neg_files_per_pos_file

        if nb_neg > nb_neg_files_per_pos_file:
            print(f"The number of negative example requested (={nb_neg}) is larger"
                  f" than the current ones available : (={nb_neg_files_per_pos_file})")
            nb_neg = nb_neg_files_per_pos_file

        # Doing some selection over files to ensure that we take the first nb_neg
        # negatives examples for a protein
        grouped_files = defaultdict(list)
        for neg_file in neg_files:
            protein_id = int(neg_file.split("_")[0])
            if len(grouped_files[protein_id]) < nb_neg:
                grouped_files[protein_id].append(neg_file)

        filtered_neg_files = sorted([file for group in grouped_files.values() for file in group])

        self.examples_files = pos_files + filtered_neg_files
        self.labels = np.array([1] * len(pos_files) + [0] * len(filtered_neg_files))

        assert len(self.labels) == len(self.examples_files)
        self.indexes = np.arange(len(self.examples_files))

        # We shuffle the data at least once
        self.shuffle()

        # Taking you some examples if asked
        if isinstance(max_examples, int) and max_examples < len(self.examples_files):
            self.indexes = self.indexes[0:max_examples]

    def nb_examples(self):
        """
        :return: the total number of examples
        """
        return len(self.indexes)

    def get_labels(self):
        """
        :return:
        """
        return self.labels[self.indexes]

    def get_examples_files(self):
        """
        :return:
        """
        return [self.examples_files[index] for index in self.indexes]

    def shuffle(self):
        """
        Shuffle the examples
        :return:
        """
        np.random.shuffle(self.indexes)

    def __len__(self):
        """
        Denotes the number of batches per epoch.

        :return:
        """
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data

        :param index: a number between 0 and self.__len__() (used to iterate in the data
        :return:
        """
        # Getting batch : the last batch can be smaller,
        # thus some book keeping around the last index
        first_index = index * self.batch_size
        last_index = min((index + 1) * self.batch_size, len(self.indexes) + 1)

        indexes = self.indexes[first_index:last_index]

        # Find list of IDs
        files_to_use = [self.examples_files[k] for k in indexes]

        # Generate data
        cubes, ys = self.__data_generation(files_to_use)

        return cubes, ys

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.

        This way we ensure to

        :return:
        """
        if self.shuffle_after_completion:
            np.random.shuffle(self.indexes)

    def __data_generation(self, files_to_use):
        """
        Return the first nb_examples cubes with their ys.

        :param files_to_use: the file to extract
        :return: list of cubes and list of their ys
        """

        cubes = []
        ys = []
        for index, ex_file in enumerate(files_to_use):
            file_name = os.path.join(self.examples_folder, ex_file)
            example = load_nparray(file_name)

            cube = make_cube(example, length_cube_side)
            y = 1 * is_positive(ex_file)

            cubes.append(cube)
            ys.append(y)

        # Conversion to np.ndarrays with the first axes used for examples
        cubes = np.array(cubes)
        ys = np.array(ys)

        # Checking consistency here
        assert (ys.shape[0] == len(files_to_use))
        assert (cubes.shape[0] == len(files_to_use))
        # Dimensions
        assert (cubes.shape[1:] == shape_cube)
        return cubes, ys


if __name__ == "__main__":
    for folder in [training_examples_folder, validation_examples_folder]:
        iterator = ExamplesIterator(folder)

        # Reverse iterator
        for i, (batch, ys) in enumerate(reversed(iterator)):
            plot_cube(batch[0])
            print("Checking size of last batch")
            assert (batch.shape[0] == iterator.nb_examples() % iterator.batch_size)
            print(i, batch.shape, np.mean(ys))
            break
