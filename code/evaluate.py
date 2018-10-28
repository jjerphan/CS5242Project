import argparse
import csv
import logging

import numpy as np
import os
import keras.backend as K
from keras.models import load_model
from collections import defaultdict

from discretization import RelativeCubeRepresentation, AbsoluteCubeRepresentation
from examples_iterator import ExamplesIterator
from pipeline_fixtures import get_parameters_dict
from settings import VALIDATION_EXAMPLES_FOLDER, METRICS_FOR_EVALUATION, RESULTS_FOLDER, LENGTH_CUBE_SIDE, \
    PARAMETERS_FILE_NAME_SUFFIX, EVALUATION_LOGS_FOLDER, EVALUATION_CSV_FILE
from train_cnn import f1


def mean_pred(y_pred, y_true):
    return K.mean(y_pred)


def evaluate(serialized_model_path, max_examples=None):
    """
    Evaluate a given model using custom metrics.

    List of metrics are evaluated using validation data. Evaluation results are saved into evaluation log file.

    Saves a log in its associated results folder.

    :param serialized_model_path: where the serialized_model is
    :param max_examples: the maximum number of examples to use
    :return:
    """

    # Formatting fixtures
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Making a folder for the job to save log, model, history in it.
    id = serialized_model_path.split(os.sep)[-2]
    job_folder = os.path.join(RESULTS_FOLDER, id)
    if not (os.path.exists(RESULTS_FOLDER)):
        print(f"The {RESULTS_FOLDER} does not exist. Creating it.")
        os.makedirs(RESULTS_FOLDER)

    # For final CSV file
    if not(os.path.exists(EVALUATION_LOGS_FOLDER)):
        os.makedirs(EVALUATION_LOGS_FOLDER)

    fh = logging.FileHandler(os.path.join(job_folder,  f"evaluate.log"))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.debug(f"Evaluating model: {serialized_model_path}")

    model = load_model(serialized_model_path, custom_objects={"mean_pred": mean_pred, "f1": f1})

    parameters = get_parameters_dict(job_folder=job_folder)

    cube_representation = AbsoluteCubeRepresentation(length_cube_side=LENGTH_CUBE_SIDE) \
        if parameters["representation"] == AbsoluteCubeRepresentation.name \
        else RelativeCubeRepresentation(length_cube_side=LENGTH_CUBE_SIDE)

    logger.debug(f"Representation: {cube_representation.name}")

    validation_examples_iterator = ExamplesIterator(representation=cube_representation,
                                                    examples_folder=VALIDATION_EXAMPLES_FOLDER,
                                                    max_examples=max_examples,
                                                    shuffle_after_completion=False)

    logger.debug(f"Evaluating on {validation_examples_iterator.get_nb_examples()} examples")

    ys = validation_examples_iterator.get_labels()
    y_preds = model.predict_generator(validation_examples_iterator)
    # Rounding the prediction : using the second one
    y_rounded = np.array([1 if y > 0.5 else 0 for y in y_preds])

    logger.debug("Computing metrics")
    metrics_results = dict(map(lambda metric: (metric.__name__, metric(ys, y_rounded)), METRICS_FOR_EVALUATION))

    # Gathering all the info together
    log = defaultdict(str, metrics_results)
    log["id"] = id

    for param, value in parameters.items():
        log[param] = value

    log["positives_prediction"] = len(list(filter(lambda y: y != 0, y_rounded)))
    log["negatives_prediction"] = len(list(filter(lambda y: y == 0, y_rounded)))

    logger.debug(log)
    logger.debug("Results")
    for k, v in log.items():
        logger.debug(f" {k}: {v}")

    logger.debug(f"Writting results in {EVALUATION_CSV_FILE}")

    metrics_name = list(map(lambda m: m.__name__, METRICS_FOR_EVALUATION))
    parameters_name = ["model", "nb_epochs", "nb_neg", "max_examples", "batch_size", "optimizer",
                       "representation", "weight_pos_class"]
    csv_headers = ["id", *metrics_name, "positives_prediction", "negatives_prediction", *parameters_name]

    if not(os.path.exists(EVALUATION_CSV_FILE)):
        logger.debug(f"{EVALUATION_CSV_FILE} not present ; Creating it")
        logger.debug(f"Header used {csv_headers}")
        with open(EVALUATION_CSV_FILE, "w+") as csv_fh:
            writer = csv.DictWriter(csv_fh, fieldnames=csv_headers)
            writer.writeheader()

    with open(EVALUATION_CSV_FILE, "a") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=csv_headers)
        try:
            writer.writerow(log)
        except Exception as e:
            logger.debug(f"WARNING {e}")

    logger.debug(f"Done writting results in {EVALUATION_CSV_FILE}")


if __name__ == "__main__":
    # Parsing sysargv arguments
    parser = argparse.ArgumentParser(description='Evaluate a model using a serialized version of it.')

    parser.add_argument('--model_path', metavar='model_path',
                        type=str, required=True,
                        help=f'where the serialized file of the model (.h5) is.')

    parser.add_argument('--max_examples', metavar='max_examples',
                        type=int, default=None,
                        help='the number of total examples to use in total')

    parser.add_argument('--evaluation', metavar='evaluation',
                        type=bool, default=True,
                        help='if true: action on test data from training set')

    args = parser.parse_args()

    print("Argument parsed : ", args)

    evaluate(serialized_model_path=args.model_path,
             max_examples=args.max_examples)
