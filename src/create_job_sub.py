import os
import textwrap
from models import models_available, models_available_names
from settings import job_submissions_folder, nb_neg_ex_per_pos, nb_epochs_default, batch_size_default, n_gpu_default


def save_job_file(stub, file_name):
    """
    Save a submission file.

    The stub is inserted in the given file.

    :param stub: the string that will be used as the contain of the file
    :param file_name: the name of the file to use
    :return:
    """
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
    Ask for parameters incrementally to create a job to train a given network.

    The file gets saved in `job_submissions`.

    The name of the file is made using the parameters given.
    This way we have a collection of file to submit that is easy to inspect.

    """
    script_name = "train_cnn.py"

    nb_models_available = len(models_available_names)

    # Asking for the different parameters
    model_index = -1
    while model_index not in range(nb_models_available):
        print(f"{nb_models_available} Models available:")
        for i, model in enumerate(models_available):
            print(f"  # {i}: {model.name}")

        model_index = int(input("Your choice : # "))

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

    n_gpu = input(f"Choose number of GPU (leave blank for default = {n_gpu_default}) : ")
    n_gpu = n_gpu_default if n_gpu == "" else int(n_gpu)

    assert (n_gpu > 0)

    # TODO : fix this hack to add the option
    option_max = f"\n                                                     --max_examples {max_examples} \\"

    name_job = f'train_{models_available_names[model_index]}_{nb_epochs}epochs_{batch_size}batch_{nb_neg}neg'
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
                python $PBS_O_WORKDIR/src/{script_name}  --model_index {model_index} \\
                                                         --nb_epochs {nb_epochs} \\
                                                         --batch_size {batch_size} \\
                                                         --nb_neg {nb_neg} \\
                                                         --verbose {verbose} \\{option_max if max_examples is not None else ''}
                                                         --preprocess {preprocess}
                """
    # We remove the first return in the string
    stub = stub[1:]

    file_name = os.path.join(job_submissions_folder, f"{name_job}.pbs")

    save_job_file(stub, file_name)


def create_evaluation_job():
    pass


def create_prediction_job():
    pass


if __name__ == "__main__":
    create_train_job()
