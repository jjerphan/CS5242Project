import argparse

from keras.models import load_model

from ExamplesIterator import ExamplesIterator
from create_examples import create_examples
from extraction_data import extract_data
from settings import original_predict_folder, extracted_predict_folder, predict_examples_folder, nb_neg_ex_per_pos


def predict(serialized_model_path, nb_neg, max_examples, verbose=1, preprocess=False):
    """
    1. Extract data from original pdb file and create as molecules.
    2. Mix each protein with all ligand bindings for predictions

    :param serialized_model_path: where the serialized_model is
    :param nb_neg: the number of negative examples to use per positive examples
    :param max_examples: the maximum number of examples to use
    :param verbose: to have verbose outputs
    :param preprocess: should we do some pre-processing
    :return:
    """

    print("In predict: ", serialized_model_path)

    # TODO
    # if preprocess:
    #     extract_data(original_predict_folder, is_for_training=False)
    #     create_examples(extracted_predict_folder, predict_examples_folder)
    #
    # # Load pre-trained good model
    # my_model = load_model(serialized_model_path)
    #
    # predict_examples_iterator = ExamplesIterator(examples_folder=predict_examples_folder,
    #                                              nb_neg=nb_neg,
    #                                              max_examples=max_examples)
    #
    # predictions = my_model.predict_generator(generator=predict_examples_iterator,
    #                                          verbose=verbose)
    #
    # print(predictions)

    # TODO : do some magic for maximum matching here


if __name__ == "__main__":
    # Parsing sysargv arguments
    parser = argparse.ArgumentParser(description='Predict binding of protein and ligand using a serialized model.')

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

    predict(serialized_model_path=args.model_path,
            nb_neg=args.nb_neg,
            max_examples=args.max_examples,
            verbose=args.verbose,
            preprocess=args.preprocess)
