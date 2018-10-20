import os
import textwrap

from models_inspector import ModelsInspector
from models import models_available, models_available_names
from settings import job_submissions_folder, nb_neg_ex_per_pos, nb_epochs_default, batch_size_default, n_gpu_default, \
    results_folder, name_env


def save_job_file(stub, name_job):
    """
    Save a submission file.

    The stub is inserted in the given file.

    :param stub: the string that will be used as the contain of the file
    :param name_job: the name of the job
    :return:
    """
    print("Stub inferred :")
    print(stub)

    file_name = os.path.join(job_submissions_folder, f"{name_job}.pbs")

    if input("Would you want to save the following job ? [y (default) /n]").lower() == "n":
        print("Not saved")
    else:
        # Creating the folder for job submissions if not existing
        if not (os.path.exists(job_submissions_folder)):
            os.makedirs(job_submissions_folder)

        with open(file_name, "w") as f_sub:
            # De-indenting the stub to make it in a file
            f_sub.write(textwrap.dedent(stub))

        # Showing the content of the file
        os.system(f"cat {file_name}")
        print(f"Saved in {file_name}")


def get_train_stub(model_index, name_job, nb_epochs, batch_size, nb_neg, max_examples, n_gpu):
    # TODO : fix this hack to add the option
    script_name = "train_cnn.py"

    option_max = f"\n                                                     --max_examples {max_examples} \\"

    stub = f"""
                #! /bin/bash
                #PBS -P Personal
                #PBS -q gpu
                #PBS -j oe
                #PBS -l select=1:ngpus={n_gpu}
                #PBS -l walltime=23:00:00
                #PBS -N {name_job}
                mkdir -p {results_folder}/$PBS_JOBID/
                cd $PBS_O_WORKDIR/src/
                source activate {name_env}
                python $PBS_O_WORKDIR/src/{script_name}  --model_index {model_index} \\
                                                         --nb_epochs {nb_epochs} \\
                                                         --batch_size {batch_size} \\
                                                         --nb_neg {nb_neg} \\{option_max if max_examples is not None else ''}
                                                         --job_folder {results_folder}/$PBS_JOBID/
                """
    # We remove the first return in the string
    stub = stub[1:]

    return stub


def create_train_job():
    """
    Ask for parameters incrementally to create a job to train a given network.

    The file gets saved in `job_submissions`.

    The name of the file is made using the parameters given.
    This way we have a collection of file to submit that is easy to inspect.
    """

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

    n_gpu = input(f"Choose number of GPU (leave blank for default = {n_gpu_default}) : ")
    n_gpu = n_gpu_default if n_gpu == "" else int(n_gpu)

    assert (n_gpu > 0)

    name_job = f'train_{models_available_names[model_index]}_{nb_epochs}epochs_{batch_size}batch_{nb_neg}neg'
    name_job += f"_{max_examples}max" if max_examples else ""

    stub = get_train_stub(model_index, name_job, nb_epochs, batch_size, nb_neg, max_examples, n_gpu)

    save_job_file(stub, name_job)


def create_job_with_for_one_serialized_model(script_name, name_job, evaluation=False):
    """
    The file gets saved in `job_submissions`.
    """

    # Asking for the different parameters
    model_inspector = ModelsInspector(results_folder=results_folder)
    id_model, serialized_model_path = model_inspector.choose_model()

    max_examples = input(f"Number of maximum examples to use (leave empty to use all examples) : ")
    max_examples = None if max_examples == "" else int(max_examples)

    n_gpu = input(f"Choose number of GPU (leave blank for default = {n_gpu_default}) : ")
    n_gpu = n_gpu_default if n_gpu == "" else int(n_gpu)

    # TODO : fix this hack to add the option
    option_max = f"\n                                                     --max_examples {max_examples} \\"
    option_prediction = f"                                                     --evaluation {evaluation}"

    # We append the ID for the model to it
    name_job += "_" + id_model
    assert (n_gpu > 0)

    stub = f"""
                    #! /bin/bash
                    #PBS -P Personal
                    #PBS -q gpu
                    #PBS -j oe
                    #PBS -l select=1:ngpus={n_gpu}
                    #PBS -l walltime=23:00:00
                    #PBS -N {name_job}
                    cd $PBS_O_WORKDIR/src/
                    source activate {name_env}
                    python $PBS_O_WORKDIR/src/{script_name}  --model_path {serialized_model_path} \\ {option_max if max_examples is not None else ''} \\
                                                             {option_prediction if evaluation else ''}
                    """
    # We remove the first return in the string
    stub = stub[1:]

    save_job_file(stub, name_job)


def create_multiple_train_jobs(batch_size=32, max_examples=None, n_gpu=1):
    """

    :return:
    """
    nb_models_available = len(models_available_names)

    # Asking for the different parameters
    model_index = -1
    while model_index not in range(nb_models_available):
        print(f"{nb_models_available} Models available:")
        for i, model in enumerate(models_available):
            print(f"  # {i}: {model.name}")

        model_index = int(input("Your choice : # "))

    list_nb_epochs = list(map(int, input(f"Number of epochs to use (separate with spaces then enter) : ").split()))

    list_nb_neg = list(
        map(int, input(f"Number of negatives examples to use (separate with spaces then enter) : ").split()))

    for nb_epochs in list_nb_epochs:
        for nb_neg in list_nb_neg:
            name_job = f'train_{models_available_names[model_index]}_{nb_epochs}epochs_{batch_size}batch_{nb_neg}neg'
            name_job += f"_{max_examples}max" if max_examples else ""

            stub = get_train_stub(model_index=model_index,
                                  name_job=name_job,
                                  nb_epochs=nb_epochs,
                                  batch_size=batch_size,
                                  nb_neg=nb_neg,
                                  max_examples=max_examples,
                                  n_gpu=n_gpu)

            save_job_file(stub, name_job)


def create_evaluation_job():
    """
    Prompt to create a submission file to evaluate a given model.

    :return:
    """
    create_job_with_for_one_serialized_model(script_name="evaluate.py",
                                             name_job="evaluate")


def create_prediction_job():
    """
    Prompt to create a submission file to predict using a given model.

    :return:
    """
    choice = input(f"Testing? Choose 'n' for prediction. [y (default)/n] : ")
    evaluation = False if choice == "" or choice == "y" else True
    create_job_with_for_one_serialized_model(script_name="predict.py",
                                             name_job="predict",
                                             evaluation=evaluation)


if __name__ == "__main__":
    # Choosing the time of job to create
    choice = -1
    jobs = [create_train_job, create_multiple_train_jobs, create_evaluation_job, create_prediction_job]
    while choice not in range(len(jobs)):
        for i, job in enumerate(jobs):
            print(i, job.__name__)
        choice = int(input("Your choice : # "))

    jobs[choice]()
