import os
from collections import defaultdict

from settings import history_fie_name, serialized_model_file_name, parameters_file_name


class ModelsInspector:
    """
    To iterate and access content of models.py sub-folders.

    """

    def __init__(self, models_folders):
        self.general_folder = models_folders
        self.sub_folders = list(map(lambda sub_folder: os.path.join(models_folders, sub_folder),
                                    os.listdir(models_folders)))
        serialized_models_file_names = defaultdict(str)
        histories_file_names = defaultdict(str)

        def str_default_dict():
            return defaultdict(str)

        sets_parameters = defaultdict(str_default_dict)

        for folder in self.sub_folders:
            for file in os.listdir(folder):
                if file == history_fie_name:
                    histories_file_names[folder] = file

                if file == serialized_model_file_name:
                    serialized_models_file_names[folder] = file

                if file == parameters_file_name:
                    with open(os.path.join(models_folders, folder, file), "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            words = line.split("=")
                            key = words[0]
                            value = words[1]
                            sets_parameters[folder][key] = value

                    sets_parameters[folder] = file

        # Each of those are default dict from folders to values
        # Histories are serialized model
        self.sets_parameters = sets_parameters
        self.serialized_models_file_names = serialized_models_file_names
        self.histories_file_names = histories_file_names

    def __join_path(self, folder, attribut):
        return os.path.join(self.general_folder, folder, attribut[folder])

    def get_serialized_model_file_name(self, folder):
        return self.__join_path(folder, self.serialized_models_file_names)

    def get_sets_parameters(self, folder):
        return self.sets_parameters[folder]

    def get_history_file_name(self, folder):
        return self.__join_path(folder, self.histories_file_names)

    def __len__(self):
        return len(self.sub_folders)

    def __getitem__(self, index):
        folder = self.sub_folders[index]
        return folder, self.get_sets_parameters(folder), self.get_serialized_model_file_name(folder), self.get_history_file_name(folder)
