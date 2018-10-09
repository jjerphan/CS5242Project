from keras.models import load_model
from settings import original_predict_folder, extracted_predict_folder, predict_examples_folder

from extraction_data import extract_data
from create_examples import create_examples
from pipeline_fixtures import ExamplesIterator


def predict(serialized_model, nb_neg, max_examples, verbose=1, preprocess=False):
    """
    1. Extract data from original pdb file and create as molecules.
    2. Mix each protein with all ligand bindings for predictions

    :param serialized_model:
    :param nb_neg:
    :param max_examples:
    :param verbose:
    :param preprocess:
    :return:
    """

    if preprocess:
        extract_data(original_predict_folder, is_for_training=False)
        create_examples(extracted_predict_folder, predict_examples_folder)

    # Load pre-trained good model
    my_model = load_model(serialized_model)

    predict_examples_iterator = ExamplesIterator(examples_folder=predict_examples_folder,
                                                 nb_neg=nb_neg,
                                                 max_examples=max_examples)

    predictions = my_model.predict_generator(generator=predict_examples_iterator,
                                             verbose=verbose)

    print(predictions)

    # TODO : do some magic for maximum matching here


if __name__ == '__main__':
    predict('../models/model2018-10-07_17:35:11.438784+08:00first_simple_model.h5')
