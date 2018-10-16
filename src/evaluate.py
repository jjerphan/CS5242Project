import argparse
import logging

import numpy as np
import os
from keras.models import load_model
import keras.backend as K

from examples_iterator import ExamplesIterator
from settings import testing_examples_folder, nb_neg_ex_per_pos, metrics_for_evaluation, results_folder


def mean_pred(y_pred,y_true):
    return K.mean(y_pred)


def evaluate(serialized_model_path, nb_neg, max_examples, verbose=1, preprocess=False):
    """
    Evaluate a given model using custom metrics.

    :param serialized_model_path: where the serialized_model is
    :param nb_neg: the number of negative examples to use per positive examples
    :param max_examples: the maximum number of examples to use
    :param verbose: to have verbose outputs
    :param preprocess: should we do some pre-processing
    :return:
    """

    # Formatting fixtures
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logfile = f"evaluate.log"

    # Making a folder for the job to save log, model, history in it.
    id = serialized_model_path.split(os.sep)[-2]
    job_folder = os.path.join(results_folder, id)
    if not (os.path.exists(results_folder)):
        print(f"The {results_folder} does not exist. Creating it.")
        os.makedirs(results_folder)

    fh = logging.FileHandler(os.path.join(job_folder, logfile))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.debug(f"Evaluating model: {serialized_model_path}")

    model = load_model(serialized_model_path, custom_objects={"mean_pred": mean_pred})

    test_examples_iterator = ExamplesIterator(examples_folder=testing_examples_folder,
                                              max_examples=max_examples,
                                              shuffle_after_completion=False)

    logger.debug(f"Evaluating on {test_examples_iterator.nb_examples()} examples")

    ys = test_examples_iterator.get_labels()
    y_preds = model.predict_generator(test_examples_iterator)
    # Rounding the prediction : using the second one
    y_rounded = np.array([1 if y > 0.5 else 0 for y in y_preds])

    logger.debug("Computing metrics")
    metrics_results = dict(map(lambda metric: (metric.__name__, metric(ys, y_rounded)), metrics_for_evaluation))

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

    parser.add_argument('--nb_neg', metavar='nb_neg',
                        type=int, default=nb_neg_ex_per_pos,
                        help='the number of negatives examples to use per positive example')

    parser.add_argument('--max_examples', metavar='max_examples',
                        type=int, default=None,
                        help='the number of total examples to use in total')

    parser.add_argument('--verbose', metavar='verbose',
                        type=int, default=True,
                        help='the number of total examples to use in total')

    parser.add_argument('--preprocess', metavar='preprocess',
                        type=int, default=True,
                        help='if !=0 triggers the pre-processing of the data')

    args = parser.parse_args()

    print("Argument parsed : ", args)

    assert (args.nb_neg > 0)
    assert (args.nb_neg > 0)

    evaluate(serialized_model_path=args.model_path,
             nb_neg=args.nb_neg,
             max_examples=args.max_examples,
             verbose=args.verbose,
             preprocess=args.preprocess)
