import logging
import csv

import numpy as np
import os
from keras.models import load_model
import keras.backend as K
from collections import defaultdict

from discretization import RelativeCubeRepresentation
from models_inspector import ModelsInspector
from examples_iterator import ExamplesIterator
from pipeline_fixtures import get_current_timestamp
from settings import VALIDATION_EXAMPLES_FOLDER, METRICS_FOR_EVALUATION, RESULTS_FOLDER, LENGTH_CUBE_SIDE, \
    EVALUATION_LOGS_FOLDER
from train_cnn import f1


def mean_pred(y_pred, y_true):
    return K.mean(y_pred)


def evaluate_all(max_examples=None):
    """
    Evaluate all the models that are present in the `results_folder`.
    Can take quite a long time.

    :param max_examples: the maximum number of examples to use
    :return:
    """

    # Formatting fixtures
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    current_timestamp = get_current_timestamp()
    logfile = f"evaluate_all{current_timestamp}.log"
    results_csv_file = f"evaluate_all{current_timestamp}.csv"

    # Making a folder for the job to save log, model, history in it.
    if not (os.path.exists(RESULTS_FOLDER)):
        print(f"The {RESULTS_FOLDER} does not exist. Creating it.")
        os.makedirs(RESULTS_FOLDER)

    if not (os.path.exists(EVALUATION_LOGS_FOLDER)):
        print(f"The {EVALUATION_LOGS_FOLDER} does not exist. Creating it.")
        os.makedirs(EVALUATION_LOGS_FOLDER)

    fh = logging.FileHandler(os.path.join(EVALUATION_LOGS_FOLDER, logfile))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    models_inspector = ModelsInspector(results_folder=RESULTS_FOLDER)

    logger.debug(f"Evaluating {len(models_inspector)} models")

    cube_representation = RelativeCubeRepresentation(length_cube_side=LENGTH_CUBE_SIDE)
    validation_examples_iterator = ExamplesIterator(representation=cube_representation,
                                                    examples_folder=VALIDATION_EXAMPLES_FOLDER,
                                                    max_examples=max_examples,
                                                    shuffle_after_completion=False)

    ys = validation_examples_iterator.get_labels()

    logger.debug(f"Evaluating on {validation_examples_iterator.get_nb_examples()} examples")

    # Constructing the header : we are saving the results of the evaluation with for each models
    # the parameters that have been used to train
    metrics_name = list(map(lambda m: m.__name__, METRICS_FOR_EVALUATION))
    parameters_name = ["model", "nb_epochs", "nb_neg", "max_examples", "batch_size", "optimizer"]

    headers = ["id", *metrics_name, "positives_prediction",
               "negatives_prediction", *parameters_name]

    with open(os.path.join(EVALUATION_LOGS_FOLDER, results_csv_file), "w+") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=headers)
        writer.writeheader()
        for subfolder, set_parameters, serialized_model_path, history_file_name, _ in models_inspector:

            logger.debug(f"Evaluating {subfolder}")
            logger.debug(f"Parameters")
            for k, v in set_parameters.items():
                logger.debug(f" {k}: {v}")

            model = load_model(serialized_model_path, custom_objects={"mean_pred": mean_pred, "f1": f1})

            y_preds = model.predict_generator(validation_examples_iterator)
            # Rounding the prediction : using the second one
            y_rounded = np.array([1 if y > 0.5 else 0 for y in y_preds])

            logger.debug("Computing metrics")
            metrics_results = dict(map(lambda metric: (metric.__name__, metric(ys, y_rounded)), METRICS_FOR_EVALUATION))
            log = defaultdict(str, metrics_results)

            # Gathering all the info together
            try:
                for param in parameters_name:
                    log[param] = set_parameters[param]
            except Exception as e:
                logger.debug(f"WARNING {e}")

            log["positives_prediction"] = len(list(filter(lambda y: y != 0, y_rounded)))
            log["negatives_prediction"] = len(list(filter(lambda y: y == 0, y_rounded)))

            log["id"] = subfolder.split(os.sep)[-1]

            logger.debug(log)
            logger.debug("Results")
            for k, v in log.items():
                logger.debug(f" {k}: {v}")

            logger.debug(f"Write results in {results_csv_file}")
            try:
                writer.writerow(log)
            except Exception as e:
                logger.debug(f"WARNING {e}")


if __name__ == "__main__":
    evaluate_all(max_examples=None)
