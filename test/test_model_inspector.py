import unittest
import os
import warnings

warnings.simplefilter("ignore")

from code.models_inspector import ModelsInspector


class ModelsInspectorTest(unittest.TestCase):
    """
    Testing the ModelsInspector.

    """

    def setUp(self):
        self.test_folder = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, "fixtures_results"))
        self.models_inspector = ModelsInspector(results_folder=self.test_folder)

    def test_good_folder(self):
        """
        Tests that it read just one folder (the one with the model)
        and that the info present in there are correct (especially the settings).

        :return:
        """

        self.assertEqual(len(self.models_inspector), 1)

        sub_folder, set_parameters, serialized_model_path, history_file_path, _ = self.models_inspector[0]
        true_sub_folder = os.path.join(self.test_folder, "with_serialized_model")
        self.assertEqual(sub_folder, true_sub_folder)
        self.assertEqual(serialized_model_path, os.path.join(sub_folder, "20181224122334model.h5"))
        self.assertEqual(history_file_path, os.path.join(sub_folder, "20181224122334history.pickle"))
        true_set_parameters = {
            "model_name": "pafnucy_like",
            "nb_epochs": "20",
            "max_examples": "None",
            "batch_size": "32",
            "nb_neg": "10",
            "verbose": "True",
            "preprocess": "False",
            "optimizer": "rmsprop"
        }
        self.assertEqual(true_set_parameters, set_parameters)
