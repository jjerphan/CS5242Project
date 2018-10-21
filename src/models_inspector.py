import os
from collections import defaultdict

from settings import parameters_file_name, results_folder, serialized_model_file_name_end, history_file_name_end


class ModelsInspector:
    """
    To iterate and access content of results sub-folders.

    This class exposes a way to get:
        - the path to the folder
        - the set of parameters used as a dict (parameters are extracted from `parameters.txt`.
        - the path of serialized models `model.h5`
        - the path of the history of training `history.pickle`

    Only results folders that have a serialized model in it are shown.
    This is because we are interested to study model that have been created only.
    We call those folder "inspectable".


    """

    def __init__(self, results_folder):
        self._general_folder = results_folder
        sub_folders = list(map(lambda sub_folder: os.path.join(self._general_folder, sub_folder) \
            if os.path.isdir(os.path.join(self._general_folder, sub_folder)) else None, \
                               os.listdir(self._general_folder)))
        sub_folders.append(self._general_folder)
        sub_folders = [x for x in sub_folders if x is not None]

        # It is possible that there exist sub-folders with no serialized model
        # (if the model is being trained for example) so, we chose here to
        # only keep sub folders that contains one.
        self._sub_folders = list(filter(lambda folder: [serialized_model_file_name_end in file for file in os.listdir(folder)], sub_folders))
        serialized_models_file_names = defaultdict(str)
        histories_file_names = defaultdict(str)

        def str_default_dict():
            return defaultdict(str)

        sets_parameters = defaultdict(str_default_dict)

        for folder in self._sub_folders:
            files_present = os.listdir(folder)

            for file in files_present:
                if history_file_name_end in file:
                    histories_file_names[folder] = file

                if serialized_model_file_name_end in file:
                    serialized_models_file_names[folder] = file

                if file == parameters_file_name:
                    with open(os.path.join(self._general_folder, folder, file), "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            words = line.replace("\n","").split("=")
                            key = words[0]
                            value = words[1]
                            sets_parameters[folder][key] = value

        # Each of those are default dict from folders to values
        # note that sets_parameters is a dictionary of dictionaries
        self._sets_parameters = sets_parameters
        self._serialized_models_file_names = serialized_models_file_names
        self._histories_file_names = histories_file_names

    def __join_path(self, folder, attribut):
        """
        Fixture to return the absolute path to of an attribute of a given folder
        :return:
        """
        return os.path.join(self._general_folder, folder, attribut[folder])

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
        return sub_folder, set_parameters, serialized_model_path, history_file_path

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
            for index, (folder, set_parameters, serialized_model_path, history) in enumerate(self):
                print(f"#{index} Name : {set_parameters['model_name']} (from {folder})")
                for key, value in set_parameters.items():
                    print(f" - {key}: {value}")

            model_index = int(input("Your choice : # "))

        sub_folder, _, serialized_model_path, _ = self[model_index]

        # Using the name sub folder as an identifier for now
        id = sub_folder.split(os.sep)[-1]

        return id, serialized_model_path


if __name__ == "__main__":
    model_inspector = ModelsInspector(results_folder=results_folder)

    print(model_inspector[0])
