import os
from keras.models import load_model
from settings import original_predict_folder, extracted_predict_folder, predict_examples_folder

from extraction_data import extract_data
from create_examples import create_examples
from pipeline_fixtures import Training_Example_Iterator
from train_cnn import mean_pred

def predict(model):
    # Preprocessing - 1. Extract data from original pdb file and create as molecues.
    #                 2. Mix each protein with all ligand bindings for prediction
    extract_data(original_predict_folder, split_training=False)
    create_examples(extracted_predict_folder, predict_examples_folder)

    # Load pre-trained good model
    my_model = load_model(model, custom_objects={'mean_pred': mean_pred})

    #complexes = os.listdir(predict_examples_folder)

    predict_examples_iterator = Training_Example_Iterator(examples_folder=predict_examples_folder,
                                                       nb_neg=9,
                                                       max_examples=10)


    prediction = my_model.predict_generator(generator=predict_examples_iterator,
                               verbose=1)

    print(prediction)

if __name__ == '__main__':
    predict('../models/model2018-10-07_17:35:11.438784+08:00first_simple_model.h5')