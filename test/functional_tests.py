import unittest
import os
import warnings

warnings.simplefilter("ignore")

import shutil

from code.models_inspector import ModelsInspector
from code.pipeline_fixtures import get_current_timestamp
from code.settings import SERIALIZED_MODEL_FILE_NAME_SUFFIX, PARAMETERS_FILE_NAME_SUFFIX, \
    TRAINING_LOGFILE_SUFFIX, HISTORY_FILE_NAME_SUFFIX
from code.train_cnn import train_cnn


class TestTrainingJob(unittest.TestCase):

    def setUp(self):
        self.test_folder = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, get_current_timestamp()))
        os.makedirs(self.test_folder)

    def tearDown(self):
        shutil.rmtree(self.test_folder)

    def test_train_cnn(self):
        """
        Testing the training job of the pipeline:
            - The model should be trained
            - Results should be saved in the right place
            - ModelsInspector should detected the trained model then

        """

        list_res = os.listdir(self.test_folder)
        self.assertEqual(len(list_res), 0)

        nb_trials = 3
        for i in range(nb_trials):
            train_cnn(model_index=0,
                      nb_epochs=1,
                      nb_neg=1,
                      max_examples=2,
                      batch_size=2,
                      results_folder=self.test_folder)

        list_res = os.listdir(self.test_folder)
        self.assertEqual(len(list_res), nb_trials)

        should_be_saved = sorted(
            [PARAMETERS_FILE_NAME_SUFFIX, SERIALIZED_MODEL_FILE_NAME_SUFFIX, HISTORY_FILE_NAME_SUFFIX,
             TRAINING_LOGFILE_SUFFIX])

        file_saved = sorted(os.listdir(os.path.join(self.test_folder, list_res[0])))

        files_presents = map(lambda f: any([f in file for file in file_saved]), should_be_saved)

        self.assertTrue(all(files_presents))

        models_inspector = ModelsInspector(results_folder=self.test_folder)

        self.assertEqual(len(models_inspector), nb_trials)
