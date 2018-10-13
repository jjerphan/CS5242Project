import os
from keras.models import load_model
from settings import original_predict_folder, extracted_predict_folder, predict_examples_folder

from extraction_data import extract_data
from create_examples import create_examples
from ExamplesIterator import ExamplesIterator

def predict(model):
    # Preprocessing - 1. Extract data from original pdb file and create as molecues.
    #                 2. Mix each protein with all ligand bindings for prediction
    def pre_processing():
        extract_data(original_predict_folder, False)
        create_examples(extracted_predict_folder, predict_examples_folder)

    # Load pre-trained good model
    my_model = load_model(model)

    #complexes = os.listdir(predict_examples_folder)

    predict_examples_iterator = ExamplesIterator(examples_folder=predict_examples_folder,
                                                       nb_neg=9,
                                                       max_examples=10)


    prediction = my_model.predict_generator(generator=predict_examples_iterator,
                               verbose=1)

    print(prediction)

if __name__ == '__main__':
    predict('../models/model2018-10-07_17:35:11.438784+08:00first_simple_model.h5')