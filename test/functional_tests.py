import unittest
import os
import warnings
warnings.simplefilter("ignore")

import shutil

from ModelsInspector import ModelsInspector
from pipeline_fixtures import get_current_timestamp
from settings import history_file_name, serialized_model_file_name, parameters_file_name, \
    training_logfile
from train_cnn import train_cnn


class TestTrainingJob(unittest.TestCase):
    """
    Testing the training job of the pipeline.
    """

    def setUp(self):
        self.test_folder = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, get_current_timestamp()))
        os.makedirs(self.test_folder)

    def tearDown(self):
        shutil.rmtree(self.test_folder)

    def test_train_cnn(self):
        """
        Should not fails and should save results in different a folders.
        """

        list_res = os.listdir(self.test_folder)
        self.assertEqual(len(list_res), 0)

        nb_trials = 3
        for i in range(nb_trials):
            train_cnn(model_index=0,
                      nb_epochs=1,
                      nb_neg=1,
                      max_examples=2,
                      verbose=1,
                      preprocess=0,
                      batch_size=2,
                      results_folder=self.test_folder)

        list_res = os.listdir(self.test_folder)
        self.assertEqual(len(list_res), nb_trials)

        should_be_saved = sorted(
            [parameters_file_name, serialized_model_file_name, history_file_name, training_logfile])
        file_saved = sorted(os.listdir(os.path.join(self.test_folder, list_res[0])))

        self.assertEqual(should_be_saved, file_saved)

        models_inspector = ModelsInspector(results_folder=self.test_folder)

        self.assertEqual(len(models_inspector), nb_trials)
