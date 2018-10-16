import pickle
import logging
import os
import argparse

from datetime import datetime

import keras
from keras.callbacks import EarlyStopping
from keras.losses import binary_crossentropy

from pipeline_fixtures import LogEpochBatchCallback, get_current_timestamp
from examples_iterator import ExamplesIterator
from models import models_available, models_available_names
from settings import training_examples_folder, testing_examples_folder, results_folder, nb_neg_ex_per_pos, \
    optimizer_default, batch_size_default, nb_epochs_default, original_data_folder, \
    extracted_data_train_folder, extracted_data_test_folder, serialized_model_file_name, history_file_name,\
    parameters_file_name, training_logfile
from extraction_data import extract_data
from create_examples import create_examples


def train_cnn(model_index, nb_epochs, nb_neg, max_examples, verbose, preprocess, batch_size,
              optimizer=optimizer_default, results_folder=results_folder, job_folder=None):
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
    :param results_folder: where to save results
    :return:
    """
    # Formatting fixtures
    current_timestamp = get_current_timestamp()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Making a folder for the job to save log, model, history in it.
    if job_folder is None:
        job_folder = os.path.join(results_folder, current_timestamp)
    if not (os.path.exists(results_folder)):
        print(f"The {results_folder} does not exist. Creating it.")
        os.makedirs(results_folder)

    # Creating the folder for the job
    if not (os.path.exists(job_folder)):
        os.makedirs(job_folder)

    fh = logging.FileHandler(os.path.join(job_folder, training_logfile))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    start_time = datetime.now()

    # Eventual pre-processing
    if preprocess:
        logger.debug('Extracting data.')
        extract_data(original_data_folder)
        print('Creating training examples')
        create_examples(extracted_data_train_folder, training_examples_folder, nb_neg_ex_per_pos)
        print('Creating testing examples')
        create_examples(extracted_data_test_folder, testing_examples_folder, nb_neg_ex_per_pos)

    preprocessing_checkpoint = datetime.now()

    logger.debug('Creating network model')
    model = models_available[model_index]
    logger.debug(f"Model {model.name} chosen")
    keras.utils.print_summary(model, print_fn=logger.debug)

    model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=['accuracy'])

    logger.debug(f'{os.path.basename(__file__)} : Training the model with the following parameters')
    logger.debug(f'model = {model.name}')
    logger.debug(f'nb_epochs   = {nb_epochs}')
    logger.debug(f'max_examples   = {max_examples}')
    logger.debug(f'batch_size  = {batch_size}')
    logger.debug(f'nb_neg      = {nb_neg}')
    logger.debug(f'verbose     = {verbose}')
    logger.debug(f'preprocess  = {preprocess}')
    logger.debug(f'optimizer   = {optimizer}')

    with open(os.path.join(job_folder, parameters_file_name), "w") as f:
        f.write(f'model={model.name}\n')
        f.write(f'nb_epochs={nb_epochs}\n')
        f.write(f'max_examples={max_examples}\n')
        f.write(f'batch_size={batch_size}\n')
        f.write(f'nb_neg={nb_neg}\n')
        f.write(f'verbose={verbose}\n')
        f.write(f'preprocess={preprocess}\n')
        f.write(f'optimizer={optimizer}\n')

    logger.debug(f'model, log and history to be saved in {job_folder}')

    # To load the data incrementally
    train_examples_iterator = ExamplesIterator(examples_folder=training_examples_folder,
                                               nb_neg=nb_neg,
                                               batch_size=batch_size,
                                               max_examples=max_examples)

    test_examples_iterator = ExamplesIterator(examples_folder=testing_examples_folder,
                                              nb_neg=nb_neg,
                                              batch_size=batch_size,
                                              max_examples=max_examples)

    # To log batches and epoch
    epoch_batch_callback = LogEpochBatchCallback(logger)
    # earlystopping = EarlyStopping(patience=1)

    # To prevent having
    class_weight = {
                     0: 1,
                     1: nb_neg
                     }

    logger.debug(f'Training with class_weight: {class_weight}')

    # Here we go !
    history = model.fit_generator(generator=train_examples_iterator,
                                  epochs=nb_epochs,
                                  verbose=verbose,
                                  validation_data=test_examples_iterator,
                                  callbacks=[epoch_batch_callback],
                                  class_weight=class_weight)

    logger.debug('Done training !')
    train_checkpoint = datetime.now()

    # Saving models.py and history
    model_file = os.path.join(job_folder, serialized_model_file_name)
    history_file = os.path.join(job_folder, history_file_name)

    model.save(model_file)
    logger.debug(f"Model saved in {model_file}")
    with open(history_file, "wb") as handle:
        pickle.dump(history.history, handle)

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
                        help='if !=0 triggers the pre-processing of the data')

    parser.add_argument('--job_folder', metavar='job_folder',
                        type=str, default=True,
                        help='the folder where results are to be saved')

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
              preprocess=args.preprocess,
              job_folder=args.job_folder)
