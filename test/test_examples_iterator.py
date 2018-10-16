import unittest
import numpy as np
import warnings
warnings.simplefilter("ignore")

from src.examples_iterator import ExamplesIterator
from src.pipeline_fixtures import is_positive
from src.settings import training_examples_folder


class ExamplesIteratorTest(unittest.TestCase):
    """
    Testing the ExamplesIterator.

    """

    def test_good_indexing(self):
        """
        Test if the indexing remain consistent

        :return:
        """

        training_examples_iterator = ExamplesIterator(examples_folder=training_examples_folder)

        labels = training_examples_iterator.get_labels()
        examples_files = training_examples_iterator.get_examples_files()

        indexed_labels = np.array([1 if is_positive(file) else 0 for file in examples_files])

        np.testing.assert_array_equal(labels, indexed_labels)

        # Shuffling examples
        training_examples_iterator.shuffle()

        labels_after_shuffle = training_examples_iterator.get_labels()
        examples_files_after_shuffle = training_examples_iterator.get_examples_files()

        indexed_labels_after_shuffle = np.array(
            [1 if is_positive(file) else 0 for file in examples_files_after_shuffle])

        np.testing.assert_array_equal(labels_after_shuffle, indexed_labels_after_shuffle)
