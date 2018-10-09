import os
from collections import defaultdict

import keras
import numpy as np

from discretization import load_nparray, make_cube
from settings import extract_id, progress, extracted_protein_suffix, extracted_ligand_suffix, resolution_cube, \
    training_examples_folder, nb_channels, nb_neg_ex_per_pos


def is_positive(name):
    """
    name is of the form "xxxx_yyyy[.csv]"

    'xxxx' corresponds to the protein.
    'yyyy' corresponds to the ligands

    Return xxxx == yyyy (both molecules bind together)

    :param name: the name of a file of the form "xxxx_yyyy[.csv]"
    :return:
    """
    systems = name.replace(".csv", "").split("_")
    return systems[0] == systems[1]


def is_negative(name):
    """
    name is of the form "xxxx_yyyy[.csv]"

    'xxxx' corresponds to the protein.
    'yyyy' corresponds to the ligands

    Return xxxx != yyyy (both molecules don't bind together)

    :param name: the name of a file of the form "xxxx_yyyy[.csv]"
    :return:
    """
    return not (is_positive(name))


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

        # Taking you some examples if not
        if isinstance(max_examples, int):
            self.examples_files = self.examples_files[0:max_examples]

        self.indexes = np.arange(len(self.examples_files))

        # We shuffle the data at least once
        np.random.shuffle(self.indexes)

    def nb_examples(self):
        """
        :return: the total number of examples
        """
        return len(self.indexes)

    def __len__(self):
        """
        Denotes the number of batches per epoch.

        :return:
        """
        return int(np.ceil(len(self.examples_files) / self.batch_size))

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


class LogEpochBatchCallback(keras.callbacks.LambdaCallback):

    # From Keras documentation : https://keras.io/callbacks/
    # on_epoch_end: logs include acc and loss, and optionally include val_loss (if validation is enabled in fit)
    # and val_acc (if validation and accuracy monitoring are enabled).
    # on_batch_begin: logs include size, the number of samples in the current batch.
    # on_batch_end: logs include loss, and optionally acc (if accuracy monitoring is enabled).

    def _on_batch_begin(self, batch, logs=None):
        self.logger.debug(f"batch {batch} ; size {logs['size']}")

    def _on_batch_end(self, batch, logs=None):
        self.logger.debug("loss {:10.4f}".format(logs["loss"]))

    def _on_epoch_begin(self, epoch, logs=None):
        self.logger.debug(f"starting epoch {epoch}")

    def _on_epoch_end(self, epoch, logs=None):
        self.logger.debug(f"ending epoch {epoch}" + " ; accuracy  {:.2%} ; loss {:10.4f}".format(logs["acc"],
                                                                                                 logs["loss"]))

    def __init__(self, logger):
        self.logger = logger
        super().__init__(on_epoch_begin=self._on_epoch_begin,
                         on_epoch_end=self._on_epoch_end,
                         on_batch_begin=self._on_batch_begin,
                         on_batch_end=self._on_batch_end)


if __name__ == "__main__":
    iterator = ExamplesIterator(training_examples_folder)

    # Reverse iterator
    for i, (batch, ys) in enumerate(reversed(iterator)):
        print("Checking size of last batch")
        assert(batch.shape[0] == iterator.nb_examples() % iterator.batch_size)
        print(i, batch.shape, np.mean(ys))
        break
