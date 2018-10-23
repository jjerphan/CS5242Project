import os
import textwrap

from discretization import RelativeCubeRepresentation, AbsoluteCubeRepresentation
from models_inspector import ModelsInspector
from models import models_available, models_available_names
from settings import JOB_SUBMISSIONS_FOLDER, NB_NEG_EX_PER_POS, NB_EPOCHS_DEFAULT, BATCH_SIZE_DEFAULT, N_GPU_DEFAULT, \
    RESULTS_FOLDER, JOBS_ENV, WEIGHT_POS_CLASS


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

    file_name = os.path.join(JOB_SUBMISSIONS_FOLDER, f"{name_job}.pbs")

    if input("Would you want to save the following job ? [y (default) /n]").lower() == "n":
        print("Not saved")
    else:
        # Creating the folder for job submissions if not existing
        if not (os.path.exists(JOB_SUBMISSIONS_FOLDER)):
            os.makedirs(JOB_SUBMISSIONS_FOLDER)

        with open(file_name, "w") as f_sub:
            # De-indenting the stub to make it in a file
            f_sub.write(textwrap.dedent(stub))

        # Showing the content of the file
        os.system(f"cat {file_name}")
        print(f"Saved in {file_name}")
        print()
        print(f"To submit the job, just run in the root of the folder :")
        print(f"$ qsub {file_name}")


def get_train_stub(model_index, name_job, nb_epochs, batch_size, nb_neg, max_examples, n_gpu, weight_pos_class,
                   representation):
    """
    Return the stub for training a using the different parameters given.

    :param model_index:
    :param name_job:
    :param nb_epochs:
    :param batch_size:
    :param nb_neg:
    :param max_examples:
    :param n_gpu:
    :param weight_pos_class:
    :param representation:
    :return:
    """
    script_name = "train_cnn.py"

    option_max = f"\n                                                         --max_examples {max_examples} \\"

    stub = f"""
                #! /bin/bash
                #PBS -P Personal
                #PBS -q gpu
                #PBS -j oe
                #PBS -l select=1:ngpus={n_gpu}
                #PBS -l walltime=23:00:00
                #PBS -N {name_job}
                mkdir -p {RESULTS_FOLDER}/$PBS_JOBID/
                cd $PBS_O_WORKDIR/code/
                source activate {JOBS_ENV}
                python $PBS_O_WORKDIR/code/{script_name}  --model_index {model_index} \\
                                                         --nb_epochs {nb_epochs} \\
                                                         --batch_size {batch_size} \\
                                                         --weight_pos_class {weight_pos_class} \\
                                                         --representation {representation} \\
                                                         --nb_neg {nb_neg}\\{option_max if max_examples is not None else ''}
                                                         --job_folder {RESULTS_FOLDER}/$PBS_JOBID/
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

    nb_epochs = input(f"Number of epochs (default = {NB_EPOCHS_DEFAULT}) : ")
    nb_epochs = NB_EPOCHS_DEFAULT if nb_epochs == "" else int(nb_epochs)

    batch_size = input(f"Batch size (default = {BATCH_SIZE_DEFAULT}) : ")
    batch_size = BATCH_SIZE_DEFAULT if batch_size == "" else int(batch_size)

    nb_neg = input(f"Number of negatives examples to use (leave empty for default = {NB_NEG_EX_PER_POS}) : ")
    nb_neg = NB_NEG_EX_PER_POS if nb_neg == "" else int(nb_neg)

    max_examples = input(f"Number of maximum examples to use (leave empty to use all examples) : ")
    max_examples = None if max_examples == "" else int(max_examples)

    weight_pos_class = input(f"Weight for the positive class (leave blank for default = {WEIGHT_POS_CLASS}) : ")
    weight_pos_class = WEIGHT_POS_CLASS if weight_pos_class == "" else int(weight_pos_class)

    representation = input(f"Cube representation: leave blank for relative, type any character for absolute : ")
    representation = RelativeCubeRepresentation.name if representation == "" else AbsoluteCubeRepresentation.name

    n_gpu = input(f"Choose number of GPU (leave blank for default = {N_GPU_DEFAULT}) : ")
    n_gpu = N_GPU_DEFAULT if n_gpu == "" else int(n_gpu)

    assert (n_gpu > 0)

    name_job = f'train_{models_available_names[model_index]}_{nb_epochs}epochs_{batch_size}batch_{nb_neg}neg'
    name_job += f"_{max_examples}max" if max_examples else ""

    stub = get_train_stub(model_index, name_job, nb_epochs, batch_size, nb_neg, max_examples, n_gpu, weight_pos_class,
                          representation)

    save_job_file(stub, name_job)


def create_job_with_for_one_serialized_model(script_name, evaluation=True):
    """
    Create submission files to executed on one serialized model.

    :param script_name: the python script file name in `code/`
    :param evaluation: boolean that indicates if the job has to be created for evaluation (testing)
    :return:
    """

    # Asking for the different parameters
    model_inspector = ModelsInspector(results_folder=RESULTS_FOLDER)
    id_model, serialized_model_path = model_inspector.choose_model()

    max_examples = input(f"Number of maximum examples to use (leave empty to use all examples) : ")
    max_examples = None if max_examples == "" else int(max_examples)

    n_gpu = input(f"Choose number of GPU (leave blank for default = {N_GPU_DEFAULT}) : ")
    n_gpu = N_GPU_DEFAULT if n_gpu == "" else int(n_gpu)

    # TODO : fix this hack to add the option
    option_max = f"                                         --max_examples {max_examples} \\"

    # We append the ID for the model to it
    name_job = f"{'' if evaluation else 'final_'}{script_name.split('.')[0]}_{id_model}"
    assert (n_gpu > 0)

    stub = f"""
                    #! /bin/bash
                    #PBS -P Personal
                    #PBS -q gpu
                    #PBS -j oe
                    #PBS -l select=1:ngpus={n_gpu}
                    #PBS -l walltime=23:00:00
                    #PBS -N {name_job}
                    cd $PBS_O_WORKDIR/code/
                    source activate {JOBS_ENV}
                    python $PBS_O_WORKDIR/code/{script_name}  --model_path {serialized_model_path} \\{option_max if max_examples is not None else ''}
                                                             --evaluation {evaluation}
                    """
    # We remove the first return in the string
    stub = stub[1:]

    save_job_file(stub, name_job)


def create_multiple_train_jobs(batch_size: int=BATCH_SIZE_DEFAULT, max_examples=None, n_gpu=N_GPU_DEFAULT):
    """
    Create several submissions files to train different models.

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

    weight_pos_class = input(f"Weight for the positive class (leave blank for default = {WEIGHT_POS_CLASS}) : ")
    weight_pos_class = WEIGHT_POS_CLASS if weight_pos_class == "" else int(weight_pos_class)

    representation = input(f"Cube representation: leave blank for relative, type any character for absolute: ")
    representation = RelativeCubeRepresentation.name if representation == "" else AbsoluteCubeRepresentation.name

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
                                  n_gpu=n_gpu, weight_pos_class=weight_pos_class,
                                  representation=representation)

            save_job_file(stub, name_job)


def create_evaluation_job():
    """
    Prompt to create a submission file to evaluate a given model.

    :return:
    """
    create_job_with_for_one_serialized_model(script_name="evaluate.py", evaluation=True)


def create_prediction_job():
    """
    Prompt to create a submission file to predict using a given model.

    :return:
    """
    print("You can test a model (evaluating its performance) or predict on new using this model.")
    choice = input(f"Enter any character to predict. Leave empty for evaluation.")
    evaluation = (choice == "")
    create_job_with_for_one_serialized_model(script_name="predict.py",
                                             evaluation=evaluation)


if __name__ == "__main__":
    # Choosing the time of job to create
    choice = -1
    jobs = [create_train_job, create_multiple_train_jobs, create_evaluation_job, create_prediction_job]
    description_jobs = ["File to train just one specific model",
                        "Several files to train specific models",
                        "File to evaluate a saved model",
                        "File to predict using a saved model"]
    while choice not in range(len(jobs)):
        print("What do you want to create?")
        for i, job in enumerate(jobs):
            print(i, description_jobs[i])
        choice = int(input("Your choice : # "))

    jobs[choice]()
