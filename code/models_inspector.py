import os
from collections import defaultdict

from pipeline_fixtures import get_parameters_dict
from settings import PARAMETERS_FILE_NAME_SUFFIX, SERIALIZED_MODEL_FILE_NAME_SUFFIX, HISTORY_FILE_NAME_SUFFIX


class ModelsInspector:
    """
    To iterate and access content of results sub-folders.

    This class exposes a way to get:
        - the path to the folder
        - the set of parameters used as a dict (parameters are extracted from `parameters.txt`).
        - the path of serialized models `model.h5`
        - the path of the history of training `history.pickle`

    By default, only results folders that have a serialized model in it are shown.
    This is because we are interested to study model that have been created only.
    We call those folder "inspectable".

    All folders can be shown by setting `show_without_serialized` to True


    """

    def __init__(self, results_folder, show_without_serialized=False):
        self._general_folder = results_folder
        # Storing absolute path of sub folders
        self._sub_folders = sorted(list(filter(lambda folder: os.path.isdir(folder),
                                      map(lambda folder: os.path.join(self._general_folder, folder),
                                          os.listdir(self._general_folder)))))

        if not(show_without_serialized):
            # It is possible that there exist sub-folders with no serialized model
            # (if the model is being trained for example) so, we chose here to
            # only keep sub folders that contains one.
            contain_serialized_model = lambda folder: any([SERIALIZED_MODEL_FILE_NAME_SUFFIX in file for file in os.listdir(folder)])
            self._sub_folders = list(filter(contain_serialized_model, self._sub_folders))
        serialized_models_file_names = defaultdict(str)
        histories_file_names = defaultdict(str)
        was_evaluated = defaultdict(bool)

        def str_defaultdict():
            return defaultdict(str)

        sets_parameters = defaultdict(str_defaultdict)

        for sub_folder in self._sub_folders:
            files_present = os.listdir(sub_folder)

            for file in files_present:
                if HISTORY_FILE_NAME_SUFFIX in file:
                    histories_file_names[sub_folder] = file

                if SERIALIZED_MODEL_FILE_NAME_SUFFIX in file:
                    serialized_models_file_names[sub_folder] = file

                if PARAMETERS_FILE_NAME_SUFFIX in file:
                    sets_parameters[sub_folder] = get_parameters_dict(os.path.join(self._general_folder, sub_folder))

                if "evaluate.log" in file:
                    was_evaluated[sub_folder] = True

        # Each of those are default dict from folders to values
        # note that sets_parameters is a dictionary of dictionaries
        self._was_evaluated = was_evaluated
        self._sets_parameters = sets_parameters
        self._serialized_models_file_names = serialized_models_file_names
        self._histories_file_names = histories_file_names

    def __join_path(self, sub_folder, attribut):
        """
        Fixture to return the absolute path to of an attribute of a given folder
        :return:
        """
        return os.path.join(self._general_folder, sub_folder, attribut[sub_folder])

    def get_serialized_model_path(self, folder):
        """
        :return: the absolute path of the serialized model
        """
        return self.__join_path(folder, self._serialized_models_file_names)

    def get_sets_parameters(self, folder):
        """
        :return: the sets of parameters for one model
        """
        return self._sets_parameters[folder]

    def get_history_path(self, folder):
        """
        :return: the absolute path of the history
        """
        return self.__join_path(folder, self._histories_file_names)

    def __len__(self):
        """
        :return: the number of folders that are inspectable.
        """
        return len(self._sub_folders)

    def __getitem__(self, index):
        """
        Used for iterations.

        For one index, get the associated folder and return all the info, that are:
            - this folder path
            - the set of parameters
            - the absolute path of the serialized model
            - the absolute path of the history

        :param index:
        :return:
        """
        sub_folder = self._sub_folders[index]
        set_parameters = self.get_sets_parameters(sub_folder)
        serialized_model_path = self.get_serialized_model_path(sub_folder)
        history_file_path = self.get_history_path(sub_folder)
        return sub_folder, set_parameters, serialized_model_path, history_file_path, self._was_evaluated[sub_folder]

    def choose_model(self):
        """
        Let choose a model in the ones that exist.

        An identifier (for now the name of the folder containing the serialized model) is returned
        as well as the absolute path leading to the model.

        :return: an identifier of the model, the absolute path to the model
        """
        model_index = -1
        while model_index not in range(len(self)):
            print("Choose the model to evaluate")
            for index, (folder, set_parameters, serialized_model_path, history, was_evaluated) in enumerate(self):
                print(f"#{index} Name : {set_parameters['model_name']} (from {folder})")
                if was_evaluated:
                    print("Already evaluated")
                for key, value in set_parameters.items():
                    print(f" - {key}: {value}")

            model_index = int(input("Your choice : # "))

        sub_folder, _, serialized_model_path, _, _ = self[model_index]

        # Using the name sub folder as an identifier for now
        id = sub_folder.split(os.sep)[-1]

        return id, serialized_model_path

