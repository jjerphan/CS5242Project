import os
import textwrap
from architectures import architectures_available, architectures_available_names
from settings import job_submissions_folder, nb_neg_ex_per_pos, nb_epochs_default, batch_size_default, n_gpu_default, \
    parameters_file_name, serialized_model_file_name, history_fie_name
from collections import defaultdict

name_env = "CS5242_gpu"


class ModelInspector:
    """
    To iterate and access content of models sub-folders.

    """
    def __init__(self, models_folders):
        self.general_folder = models_folders
        self.sub_folders = list(map(lambda f: os.path.join(models_folders, f), os.listdir(models_folders)))
        serialized_models = defaultdict(str)
        histories = defaultdict(str)

        def str_default_dict(): return defaultdict(str)

        sets_parameters = defaultdict(str_default_dict)

        for folder in self.sub_folders:
            for file in os.listdir(folder):
                if file == history_fie_name:
                    histories[folder] = file

                if file == serialized_model_file_name:
                    serialized_models[folder] = file

                if file == parameters_file_name:
                    with open(os.path.join(models_folders, folder, file), "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            words = line.split("=")
                            key = words[0]
                            value = words[1]
                            sets_parameters[folder][key] = value

                    sets_parameters[folder] = file

        self.sets_parameters = sets_parameters
        self.serialized_models = serialized_models
        self.histories = histories

    def __join_path(self, index, attribut):
        return os.path.join(self.general_folder, self.sub_folders[index], attribut[index])

    def get_serialized_model(self, index):
        return self.__join_path(index,self.serialized_models)

    def get_sets_parameters(self, index):
        return self.__join_path(index, self.sets_parameters)

    def get_history(self, index):
        return self.__join_path(index, self.histories)

    def __len__(self):
        return len(self.sub_folders)

    def __getitem__(self, index):
        folder = self.sub_folders[index]
        return folder, self.get_sets_parameters(folder), self.get_serialized_model(folder), self.get_history(folder)


def save_job_file(stub, file_name):
    print("Stub inferred :")
    print(stub)

    if input("Would you want to save the following job ? [y/n (default)]").lower() != "y":
        print("Not saved")
    else:
        # Creating the folder for job submissions if not existing
        if not (os.path.exists(job_submissions_folder)):
            os.makedirs(job_submissions_folder)

        with open(file_name, "a") as f_sub:
            # De-indenting the stub to make it in a file
            f_sub.write(textwrap.dedent(stub))

        # Showing the content of the file
        os.system(f"cat {file_name}")
        print(f"Saved in {file_name}")


def create_train_job():
    """

    :return:
    """
    script_name = "train_cnn.py"

    nb_architecture_available = len(architectures_available_names)

    # Choice of architecture
    architecture_index = -1
    while architecture_index not in range(nb_architecture_available):
        print(f"{nb_architecture_available} Architecture available:")
        for i, architecture in enumerate(architectures_available):
            print(f"  # {i}: {architecture.name}")

        architecture_index = int(input("Your choice : # "))

    nb_epochs = input(f"Number of epochs (default = {nb_epochs_default}) : ")
    nb_epochs = nb_epochs_default if nb_epochs == "" else int(nb_epochs)

    batch_size = input(f"Batch size (default = {batch_size_default}) : ")
    batch_size = batch_size_default if batch_size == "" else int(batch_size)

    nb_neg = input(f"Number of negatives examples to use (leave empty for default = {nb_neg_ex_per_pos}) : ")
    nb_neg = nb_neg_ex_per_pos if nb_neg == "" else int(nb_neg)

    max_examples = input(f"Number of maximum examples to use (leave empty to use all examples) : ")
    max_examples = None if max_examples == "" else int(max_examples)

    verbose = 1 * (input(f"Keras verbose output during training? [y (default)/n] : ").lower() != "n")
    preprocess = 1 * (input(f"Extract data and create training examples? [y/n (default)] :").lower() == "y")

    # Choice
    n_gpu = input(f"Choose number of GPU (leave blank for default = {n_gpu_default}) : ")
    n_gpu = n_gpu_default if n_gpu == "" else int(n_gpu)

    assert (n_gpu > 0)

    option_max = f"\n                                                     --max_examples {max_examples} \\"

    name_job = f'train_{architectures_available_names[architecture_index]}_{nb_epochs}epochs_{batch_size}batch_{nb_neg}neg'
    name_job += f"_{max_examples}max" if max_examples else ""
    name_job += "_preprocess" if preprocess else ""

    stub = f"""
                #! /bin/bash
                #PBS -q gpu
                #PBS -o $PBS_O_WORKDIR/logs/{name_job}.o
                #PBS -e $PBS_O_WORKDIR/logs/{name_job}.e
                #PBS -l select=1:ngpus={n_gpu}
                #PBS -l walltime=23:00:00
                #PBS -N {name_job}
                cd $PBS_O_WORKDIR/src/
                source activate {name_env}
                python $PBS_O_WORKDIR/src/{script_name}  --architecture_index {architecture_index} \\
                                                         --nb_epochs {nb_epochs} \\
                                                         --batch_size {batch_size} \\
                                                         --nb_neg {nb_neg} \\
                                                         --verbose {verbose} \\{option_max if max_examples is not None else ''}
                                                         --preprocess {preprocess}
                """
    stub = stub[1:]

    file_name = os.path.join(job_submissions_folder, f"{name_job}.pbs")

    save_job_file(stub, file_name)


def create_evaluation_job():
    pass


def create_prediction_job():
    pass


if __name__ == "__main__":
    create_train_job()
