import pickle
from time import strftime, gmtime

import logging
import os

from pipeline_fixtures import Training_Example_Iterator, LogEpochBatchCallback
from models import models_available, models_available_names
from settings import training_examples_folder, logs_folder
from extraction_data import extract_data
from create_training_examples import create_training_examples
from settings import models_folders
import keras.backend as K
from keras.losses import MSE


def main(interactive=False):

    # Formatting fixtures
    current_datetime = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logfile = f"CNN-{current_datetime}.log"
    if not(os.path.exists(logs_folder)):
        print(f"The {logs_folder} does not exist. Creating it.")
        os.makedirs(logs_folder)

    fh = logging.FileHandler(os.path.join(logs_folder, logfile))
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

    model_index = -1  # we chose the last model by default
    if interactive: # we let the choice
        while model_index not in range(len(models_available_names)):

            print("Choose your model : ")
            for i, model_name in enumerate(models_available_names):
                print(f"  - #{i}: {model_name}")

            model_index = int(input("\nEnter the index of the model : # "))

    model = models_available[model_index]

    logger.debug(f"Model {model.name} chosen")
    logger.debug(model.summary())

    def mean_pred(y_true, y_pred):
        return K.mean(y_pred)

    model.compile(optimizer='rmsprop', loss=MSE, metrics=['accuracy', mean_pred])

    logger.debug('Training the model')

    # To load the data incrementally
    train_examples_iterator = Training_Example_Iterator(examples_folder=training_examples_folder,nb_neg=100)

    # To log batches and epoch
    epoch_batch_callback = LogEpochBatchCallback(logger)

    # Here we go !
    history = model.fit_generator(generator=train_examples_iterator,
                                  verbose=1)
                                  # callbacks=[epoch_batch_callback])

    logger.debug('Done training !')

    # Saving models and history
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
    main(True)
