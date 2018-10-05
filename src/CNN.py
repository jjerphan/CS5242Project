import pickle
from time import strftime, gmtime

import os

from Example_Iterator import Example_Iterator
from models import first_model, pafnucy_like
from settings import nb_neg_ex_per_pos, training_examples_folder
from pipeline_fixtures import get_cubes
from extraction_data import extract_data
from create_training_examples import create_training_examples
import logging
from settings import models_folders
import keras.backend as K
from keras.losses import MSE

# List the models here
models_available = [first_model(), pafnucy_like()]
models_available_names = list(map(lambda model: model.name, models_available))

interactive = False

def main():

    # Formatting fixtures
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("application.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Eventual pre-processing
    if interactive:
        if input("Extract data and create examples then ? [Y/(N)]").lower() == 'y':
            logger.debug('Calling module extract data.')
            print("Extracting the data")
            extract_data()
            print('Creating examples')
            create_training_examples()

    logger.debug('Creating network model')
    if interactive:
        model_index = -1
        while model_index not in range(len(models_available_names)):

            print("Choose your model : ")
            for i, model_name in enumerate(models_available_names):
                print(f"  - #{i}: {model_name}")

            model_index = int(input("\nEnter the index of the model : # "))
    else:
        model_index = -1

    model = models_available[model_index]

    logger.debug(f"Model {model.name} chosen")
    logger.debug(model.summary())

    def mean_pred(y_true, y_pred):
        return K.mean(y_pred)

    model.compile(optimizer='rmsprop', loss=MSE, metrics=['accuracy', mean_pred])
    # We are taking systems of the first 200 proteins (pos and neg example)

    # nb_examples = 3
    # logger.debug(f'Loading dataset with {nb_examples}')
    # cubes, ys = get_cubes(nb_examples)
    #
    # assert (len(cubes) == nb_examples)
    # nb_pos_examples = len(list(filter(lambda x: x == 1, ys)))
    # nb_neg_examples = len(list(filter(lambda x: x == 0, ys)))
    # logger.debug(f"{nb_pos_examples} positive examples")
    # logger.debug(f"{nb_neg_examples} negative examples")

    logger.debug('Training the model')
    # history = model.fit(cubes, ys, verbose=1)
    train_examples_iterator = Example_Iterator(examples_folder=training_examples_folder)

    history = model.fit_generator(generator=train_examples_iterator, verbose=1)

    logger.debug('Done training !')

    # Saving models and history
    current_datetime = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
    model_file = os.path.join(models_folders, "model" + current_datetime + model.name + '.h5')
    history_file = os.path.join(models_folders, "history" + current_datetime + model.name + '.pickle')

    if not(os.path.exists(models_folders)):
        logger.debug(f'The {folder} folder does not exist. Creating it.')
        os.makedirs(models_folders)

    logger.debug(f"Saving model in {model_file}")
    model.save(model_file)
    logger.debug(f"Saving history in {model_file}")
    pickle.dump(open(history, "a"), history_file)

    logger.debug("Done")


if __name__ == "__main__":
    main()
