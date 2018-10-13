import os
from collections import defaultdict

from settings import history_file_name, serialized_model_file_name, parameters_file_name, results_folder


class ModelsInspector:
    """
    To iterate and access content of results sub-folders.


    """

    def __init__(self, results_folder):
        self._general_folder = results_folder
        sub_folders = list(map(lambda sub_folder: os.path.join(results_folder, sub_folder),
                                     os.listdir(results_folder)))

        # It is possible that there exist sub-folders with no serialized model
        # (if the model is being trained for example) so, we chose here to
        # only keep sub folders that contains one.
        self._sub_folders = list(filter(lambda folder: serialized_model_file_name in os.listdir(folder), sub_folders))
        serialized_models_file_names = defaultdict(str)
        histories_file_names = defaultdict(str)

        def str_default_dict():
            return defaultdict(str)

        sets_parameters = defaultdict(str_default_dict)

        for folder in self._sub_folders:
            files_present = os.listdir(folder)

            for file in files_present:
                if file == history_file_name:
                    histories_file_names[folder] = file

                if file == serialized_model_file_name:
                    serialized_models_file_names[folder] = file

                if file == parameters_file_name:
                    with open(os.path.join(results_folder, folder, file), "r") as f:
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
        return os.path.join(self._general_folder, folder, attribut[folder])

    def get_serialized_model_file_name(self, folder):
        return self.__join_path(folder, self._serialized_models_file_names)

    def get_sets_parameters(self, folder):
        return self._sets_parameters[folder]

    def get_history_file_name(self, folder):
        return self.__join_path(folder, self._histories_file_names)

    def __len__(self):
        return len(self._sub_folders)

    def __getitem__(self, index):
        sub_folder = self._sub_folders[index]
        set_parameters = self.get_sets_parameters(sub_folder)
        serialized_model_file_name = self.get_serialized_model_file_name(sub_folder)
        history_file_name = self.get_history_file_name(sub_folder)
        return sub_folder, set_parameters, serialized_model_file_name, history_file_name

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