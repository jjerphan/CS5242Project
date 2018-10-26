import unittest
import numpy as np
import warnings

import os

from discretization import RelativeCubeRepresentation

warnings.simplefilter("ignore")

from code.examples_iterator import ExamplesIterator
from code.pipeline_fixtures import is_positive
from code.settings import TRAINING_EXAMPLES_FOLDER, LENGTH_CUBE_SIDE, VALIDATION_EXAMPLES_FOLDER, TESTING_EXAMPLES_FOLDER


class ExamplesIteratorTest(unittest.TestCase):
    """
    Testing the ExamplesIterator.

    """
    def setUp(self):
        self.folders_to_test = [TRAINING_EXAMPLES_FOLDER, VALIDATION_EXAMPLES_FOLDER, TESTING_EXAMPLES_FOLDER]
        self.repr = RelativeCubeRepresentation(length_cube_side=LENGTH_CUBE_SIDE)

    def test_good_indexing(self):
        """
        Test if the indexing remain consistent

        :return:
        """
        training_examples_iterator = ExamplesIterator(examples_folder=TRAINING_EXAMPLES_FOLDER,
                                                      representation=self.repr)

        labels = training_examples_iterator.get_labels()
        examples_files = training_examples_iterator.get_examples_files()

        indexed_labels = np.array([1 if is_positive(file) else 0 for file in examples_files])

        np.testing.assert_array_equal(labels, indexed_labels)

        # Shuffling examples
        training_examples_iterator._shuffle()

        labels_after_shuffle = training_examples_iterator.get_labels()
        examples_files_after_shuffle = training_examples_iterator.get_examples_files()

        indexed_labels_after_shuffle = np.array(
            [1 if is_positive(file) else 0 for file in examples_files_after_shuffle])

        np.testing.assert_array_equal(labels_after_shuffle, indexed_labels_after_shuffle)

    def test_example_coverage(self):
        """
        Test if the iterator catch all the example and if it returns
        them correctly.

        :return:
        """
        for folder in self.folders_to_test:
            iterator = ExamplesIterator(representation=self.repr, examples_folder=folder)
            nb_files_in_folder = len(os.listdir(folder))
            self.assertEquals(nb_files_in_folder, iterator.get_nb_examples())
            self.assertEquals(len(iterator), int(np.ceil(iterator.get_nb_examples() / iterator.get_batch_size())))

            # The last batch created should be correct: it can contains less examples
            last_batch, _ = iterator[-1]
            self.assertEquals(last_batch.shape[0] % iterator.get_batch_size(), iterator.get_nb_examples() % iterator.get_batch_size())

