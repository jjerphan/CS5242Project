import logging
import csv

import numpy as np
import os
from keras.models import load_model
import keras.backend as K
from collections import defaultdict

from models_inspector import ModelsInspector
from examples_iterator import ExamplesIterator
from pipeline_fixtures import get_current_timestamp
from settings import testing_examples_folder, nb_neg_ex_per_pos, metrics_for_evaluation, results_folder


def mean_pred(y_pred,y_true):
    return K.mean(y_pred)


def evaluate_all(nb_neg, max_examples, verbose=1, preprocess=False):
    """
    Evaluate a given model using custom metrics.

    :param nb_neg: the number of negative examples to use per positive examples
    :param max_examples: the maximum number of examples to use
    :param verbose: to have verbose outputs
    :param preprocess: should we do some pre-processing
    :return:
    """

    # Formatting fixtures
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    current_timestamp = get_current_timestamp()
    logfile = f"evaluate_all{current_timestamp}.log"
    results_csv_file = f"evaluate_all{current_timestamp}.csv"

    evaluation_logs_folder = os.path.join(results_folder, "evaluation")
    # Making a folder for the job to save log, model, history in it.
    if not (os.path.exists(results_folder)):
        print(f"The {results_folder} does not exist. Creating it.")
        os.makedirs(results_folder)

    if not (os.path.exists(evaluation_logs_folder)):
        print(f"The {evaluation_logs_folder} does not exist. Creating it.")
        os.makedirs(evaluation_logs_folder)

    fh = logging.FileHandler(os.path.join(evaluation_logs_folder, logfile))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    models_inspector = ModelsInspector(results_folder=results_folder)

    logger.debug(f"Evaluating {len(models_inspector)} models")

    test_examples_iterator = ExamplesIterator(examples_folder=testing_examples_folder,
                                              max_examples=max_examples,
                                              shuffle_after_completion=False)

    ys = test_examples_iterator.get_labels()

    logger.debug(f"Evaluating on {test_examples_iterator.nb_examples()} examples")

    # Constructing the header : we are saving the results of the evaluation with for each models
    # the parameters that have been used to train
    metrics_name = list(map(lambda m: m.__name__, metrics_for_evaluation))
    parameters_name = ["nb_epochs", "nb_neg", "max_examples", "batch_size", "optimizer"]

    headers = ["id", "model", *metrics_name, "positives_prediction",
               "negatives_prediction", *parameters_name]

    with open(os.path.join(evaluation_logs_folder, results_csv_file), "w+") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=headers)
        writer.writeheader()
        for subfolder, set_parameters, serialized_model_path, history_file_name in models_inspector:

            logger.debug(f"Evaluating {subfolder}")
            logger.debug(f"Parameters")
            for k, v in set_parameters.items():
                logger.debug(f" {k}: {v}")

            model = load_model(serialized_model_path, custom_objects={"mean_pred": mean_pred})

            y_preds = model.predict_generator(test_examples_iterator)
            # Rounding the prediction : using the second one
            y_rounded = np.array([1 if y > 0.5 else 0 for y in y_preds])

            logger.debug("Computing metrics")
            metrics_results = dict(map(lambda metric: (metric.__name__, metric(ys, y_rounded)), metrics_for_evaluation))
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
            for k,v in log.items():
                logger.debug(f" {k}: {v}")

            logger.debug(f"Write results in {results_csv_file}")
            try:
                writer.writerow(log)
            except Exception as e:
                logger.debug(f"WARNING {e}")


if __name__ == "__main__":
    evaluate_all(nb_neg=nb_neg_ex_per_pos, max_examples=None)
