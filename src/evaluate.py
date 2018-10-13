import argparse

import pandas as pd
import numpy as np
import seaborn as sn
from keras.models import load_model

from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, confusion_matrix

from matplotlib import pyplot as plt

from ExamplesIterator import ExamplesIterator
from settings import testing_examples_folder, nb_workers, nb_neg_ex_per_pos


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

    print("In evaluate: ", serialized_model_path)

    # TODO
    # model = load_model(serialized_model_path)
    #
    # test_examples_iterator = ExamplesIterator(examples_folder=testing_examples_folder,
    #                                           shuffle_after_completion=False)
    # test_loss = model.evaluate_generator(test_examples_iterator, workers=nb_workers)
    #
    # print(f"Test loss: {test_loss}")
    #
    # ys = test_examples_iterator.get_labels()
    #
    # y_preds = model.predict_generator(test_examples_iterator)
    #
    # # Rounding the prediction : using the second one
    # y_rounded = np.array([1 if y > 0.5 else 0 for y in y_preds])
    #
    # conf_matrix = confusion_matrix(ys, y_rounded)
    #
    # df_cm = pd.DataFrame(conf_matrix, index=["Neg", "Pos"],
    #                      columns=["Neg", "Pos"])
    # plt.figure(figsize=(10, 7))
    # sn.heatmap(df_cm, annot=True)
    #
    # print(f"F1 score : {f1_score(ys,y_rounded)}")
    # print(f"Accuracy score : {accuracy_score(ys,y_rounded)}")
    # print(f"Precision score : {precision_score(ys,y_rounded)}")
    # print(f"Recall score : {recall_score(ys,y_rounded)}")
    #
    # # Counting positive predictions
    # print(len(list(filter(lambda y: y != 0, y_preds))))


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
