import os
from collections import defaultdict

import keras
import numpy as np

from discretization import load_nparray, make_cube
from pipeline_fixtures import is_positive, is_negative
from settings import nb_neg_ex_per_pos, resolution_cube, training_examples_folder


class ExamplesIterator(keras.utils.Sequence):
    """
    A class that loads data incrementally to feed Keras Model.

    The data is loaded by batch of size `batch_size`.

    We can control the amount of negative examples we can train the network on to
    make sure our network gets trained accordingly to a good distribution : need to be tweak though.

    We can choose to shuffle after each epoch or not too.

    """

    def __init__(self,
                 examples_folder,
                 batch_size=32,
                 nb_neg=nb_neg_ex_per_pos,
                 shuffle_after_completion=True,
                 max_examples: int = None):
        """

        :param examples_folder: the folder containing the examples
        :param batch_size: the batch size to use to train
        :param nb_neg: the number of positive example (this controls the number of examples
        :param shuffle_after_completion: to shuffle the data or not after each epoch
        :param max_examples: if specified, just use the number of examples given
        """

        self.batch_size = batch_size
        self.examples_folder = examples_folder
        self.shuffle = shuffle_after_completion

        if nb_neg > nb_neg_ex_per_pos:
            print(f"The number of negative example requested (={nb_neg}) is larger"
                  f" than the current one available : (={nb_neg_ex_per_pos})")

        all_files = sorted(os.listdir(examples_folder))
        pos_files = list(filter(is_positive, all_files))
        neg_files = list(filter(is_negative, all_files))

        # Doing some pre processing to ensure that we take the first nb_neg
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

        # Taking you some examples if not
        if isinstance(max_examples, int) and max_examples < len(self.examples_files):
            self.indexes = self.indexes[0:max_examples]

        # We shuffle the data at least once
        np.random.shuffle(self.indexes)

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
        if self.shuffle:
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
    iterator = ExamplesIterator(training_examples_folder)

    # Reverse iterator
    for i, (batch, ys) in enumerate(reversed(iterator)):
        print("Checking size of last batch")
        assert (batch.shape[0] == iterator.nb_examples() % iterator.batch_size)
        print(i, batch.shape, np.mean(ys))
        break
