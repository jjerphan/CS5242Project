import pickle
import logging
import os
import argparse

from time import strftime, gmtime
from datetime import datetime

from pipeline_fixtures import Training_Example_Iterator, LogEpochBatchCallback
from models import models_available, models_available_names
from settings import training_examples_folder, logs_folder, nb_neg_ex_per_pos, optimizer_default,\
    batch_size_default, nb_epochs_default
from extraction_data import extract_data
from create_training_examples import create_training_examples
from settings import models_folders
import keras.backend as K
from keras.losses import MSE

def train_cnn(model_index, nb_epochs, nb_neg, max_examples, verbose, preprocess, batch_size, optimizer=optimizer_default):
    """
    Train a given CNN.

    :param model_index: the index of the model to use in the list `model_available`
    :param nb_epochs: the number of epochs to use
    :param nb_neg: the number of training examples to use to train the network
    :param max_examples: the maximum number of examples to choose
    :param verbose: if != 0, make the output verbose
    :param preprocess: if != 0, extract the data and create the training examples
    :param batch_size: the number of examples to use per batch
    :param optimizer: the optimizer to use to train (default = "rmsprop"
    :return:
    """
    # Formatting fixtures
    current_datetime = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logfile = f"CNN-{current_datetime}.log"
    if not (os.path.exists(logs_folder)):
        print(f"The {logs_folder} does not exist. Creating it.")
        os.makedirs(logs_folder)

    fh = logging.FileHandler(os.path.join(logs_folder, logfile))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    start_time = datetime.now()

    # Eventual pre-processing
    if preprocess:
        logger.debug('Calling module extract data.')
        print("Extracting the data")
        extract_data()
        print('Creating examples')
        create_training_examples(nb_neg_ex_per_pos)

    preprocessing_checkpoint = datetime.now()

    logger.debug('Creating network model')
    model = models_available[model_index]
    logger.debug(f"Model {model.name} chosen")
    logger.debug(model.summary())

    def mean_pred(y_true, y_pred):
        return K.mean(y_pred)

    model.compile(optimizer=optimizer, loss=MSE, metrics=['accuracy', mean_pred])

    logger.debug(f'{os.path.basename(__file__)} : Training the model with the following parameters')
    logger.debug(f'model_index = {model_index}')
    logger.debug(f'nb_epochs   = {nb_epochs}')
    logger.debug(f'nb_neg      = {nb_neg}')
    logger.debug(f'verbose     = {verbose}')
    logger.debug(f'preprocess  = {preprocess}')
    logger.debug(f'optimizer   = {optimizer}')

    # To load the data incrementally
    train_examples_iterator = Training_Example_Iterator(examples_folder=training_examples_folder,
                                                        nb_neg=nb_neg,
                                                        batch_size=batch_size,
                                                        max_examples=max_examples)

    # To log batches and epoch
    epoch_batch_callback = LogEpochBatchCallback(logger)

    # Here we go !
    history = model.fit_generator(generator=train_examples_iterator,
                                  epochs=nb_epochs,
                                  verbose=verbose,
                                  callbacks=[epoch_batch_callback])

    logger.debug('Done training !')

    train_checkpoint = datetime.now()

    # Saving models and history
    model_file = os.path.join(models_folders, "model" + current_datetime + model.name + '.h5')
    history_file = os.path.join(models_folders, "history" + current_datetime + model.name + '.pickle')

    if not (os.path.exists(models_folders)):
        logger.debug(f'The {models_folders} folder does not exist. Creating it.')
        os.makedirs(models_folders)

    model.save(model_file)
    logger.debug(f"Model saved in {model_file}")
    with open(history_file, "wb") as handle:
        pickle.dump(history, handle)

    logger.debug(f"History saved in {model_file}")
    logger.debug(f"Done !")
    logger.debug(f"Preprocessing done in : {preprocessing_checkpoint - start_time}")
    logger.debug(f"Training done in      : {train_checkpoint - preprocessing_checkpoint}")


if __name__ == "__main__":
    # Parsing sysargv arguments
    parser = argparse.ArgumentParser(description='Train a neural network.')

    parser.add_argument('--model_index', metavar='model_index',
                        type=int, default=0,
                        help=f'the index of the model to use in the list {models_available_names}')

    parser.add_argument('--nb_epochs', metavar='nb_epochs',
                        type=int, default=nb_epochs_default,
                        help='the number of epochs to use')

    parser.add_argument('--batch_size', metavar='batch_size',
                        type=int, default=batch_size_default,
                        help='the number of examples to use per batch')

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
                        help='if !=0 triggers the preprocessing of the data')

    args = parser.parse_args()

    print("Argument parsed : ", args)

    assert (args.model_index < len(models_available_names))
    assert (args.nb_epochs > 0)
    assert (args.nb_neg > 0)

    train_cnn(model_index=args.model_index,
              nb_epochs=args.nb_epochs,
              nb_neg=args.nb_neg,
              max_examples=args.max_examples,
              verbose=args.verbose,
              batch_size=args.batch_size,
              preprocess=args.preprocess)
