import argparse
import logging

import numpy as np
import os
from keras.models import load_model
import keras.backend as K

from discretization import RelativeCubeRepresentation
from examples_iterator import ExamplesIterator
from settings import VALIDATION_EXAMPLES_FOLDER, METRICS_FOR_EVALUATION, RESULTS_FOLDER, LENGTH_CUBE_SIDE
from train_cnn import f1


def mean_pred(y_pred, y_true):
    return K.mean(y_pred)


def evaluate(serialized_model_path, max_examples=None):
    """
    Evaluate a given model using custom metrics.

    Saves a log in its associated results folder.

    :param serialized_model_path: where the serialized_model is
    :param max_examples: the maximum number of examples to use
    :return:
    """

    # Formatting fixtures
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logfile = f"evaluate.log"

    # Making a folder for the job to save log, model, history in it.
    id = serialized_model_path.split(os.sep)[-2]
    job_folder = os.path.join(RESULTS_FOLDER, id)
    if not (os.path.exists(RESULTS_FOLDER)):
        print(f"The {RESULTS_FOLDER} does not exist. Creating it.")
        os.makedirs(RESULTS_FOLDER)

    fh = logging.FileHandler(os.path.join(job_folder, logfile))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.debug(f"Evaluating model: {serialized_model_path}")

    model = load_model(serialized_model_path, custom_objects={"mean_pred": mean_pred, "f1": f1})

    cube_representation = RelativeCubeRepresentation(length_cube_side=LENGTH_CUBE_SIDE)
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

    metrics_results["serialized_model_path"] = serialized_model_path

    logger.debug(metrics_results)

    # Counting positive predictions
    logger.debug(len(list(filter(lambda y: y != 0, y_rounded))))


if __name__ == "__main__":
    # Parsing sysargv arguments
    parser = argparse.ArgumentParser(description='Evaluate a model using a serialized version of it.')

    parser.add_argument('--model_path', metavar='model_path',
                        type=str, required=True,
                        help=f'where the serialized file of the model (.h5) is.')

    parser.add_argument('--max_examples', metavar='max_examples',
                        type=int, default=None,
                        help='the number of total examples to use in total')

    args = parser.parse_args()

    print("Argument parsed : ", args)

    evaluate(serialized_model_path=args.model_path,
             max_examples=args.max_examples)
