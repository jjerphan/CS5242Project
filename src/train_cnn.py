import argparse
import logging
import os
import pickle
from datetime import datetime

import keras
from keras.losses import binary_crossentropy

from examples_iterator import ExamplesIterator
from models import models_available, models_available_names
from pipeline_fixtures import LogEpochBatchCallback
from pipeline_fixtures import get_current_timestamp
from settings import max_nb_neg_per_pos
from settings import training_examples_folder, results_folder, nb_neg_ex_per_pos, optimizer_default, batch_size_default, \
    nb_epochs_default, serialized_model_file_name, parameters_file_name, training_logfile, validation_examples_folder
from keras import backend as K
from keras.callbacks import EarlyStopping



def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def train_cnn(model_index, nb_epochs, nb_neg, max_examples, verbose, batch_size,
              optimizer=optimizer_default, results_folder=results_folder, job_folder=None):
    """
    Train a given CNN.

    :param model_index: the index of the model to use in the list `model_available`
    :param nb_epochs: the number of epochs to use
    :param nb_neg: the number of training examples to use to train the network
    :param max_examples: the maximum number of examples to choose
    :param verbose: if != 0, make the output verbose
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
        logger.debug(f'job_folder is None. Created timestamp based folder {job_folder}')
    if not (os.path.exists(results_folder)):
        print(f"The {results_folder} does not exist. Creating it.")
        os.makedirs(results_folder)

    # Creating the folder for the job
    if not (os.path.exists(job_folder)):
        logger.debug(f'job folder does not exist, creating {job_folder}')
        os.makedirs(job_folder)

    fh = logging.FileHandler(os.path.join(job_folder, training_logfile))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    start_time = datetime.now()

    logger.debug('Creating network model')
    model = models_available[model_index]
    logger.debug(f"Model {model.name} chosen")
    keras.utils.print_summary(model, print_fn=logger.debug)

    model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=['accuracy', f1])

    logger.debug(f'{os.path.basename(__file__)} : Training the model with the following parameters')
    logger.debug(f'model = {model.name}')
    logger.debug(f'nb_epochs   = {nb_epochs}')
    logger.debug(f'max_examples   = {max_examples}')
    logger.debug(f'batch_size  = {batch_size}')
    logger.debug(f'nb_neg      = {nb_neg}')
    logger.debug(f'optimizer   = {optimizer}')

    with open(os.path.join(job_folder, parameters_file_name), "w") as f:
        f.write(f'model={model.name}\n')
        f.write(f'nb_epochs={nb_epochs}\n')
        f.write(f'max_examples={max_examples}\n')
        f.write(f'batch_size={batch_size}\n')
        f.write(f'nb_neg={nb_neg}\n')
        f.write(f'optimizer={optimizer}\n')

    logger.debug(f'model, log and history to be saved in {job_folder}')

    # To load the data incrementally
    train_examples_iterator = ExamplesIterator(examples_folder=training_examples_folder,
                                               nb_neg=nb_neg,
                                               batch_size=batch_size,
                                               max_examples=max_examples)

    validation_examples_iterator = ExamplesIterator(examples_folder=validation_examples_folder,
                                                    nb_neg=nb_neg,
                                                    batch_size=batch_size,
                                                    max_examples=max_examples)

    # To log batches and epoch
    epoch_batch_callback = LogEpochBatchCallback(logger)
    EarlyStopping(monitor='f1', mode='max', patience=3)

    # To prevent having
    class_weight = {
        0: 1,
        1: min(nb_neg, max_nb_neg_per_pos) //2
    }

    logger.debug(f'Training with class_weight: {class_weight}')

    # Here we go !
    history = model.fit_generator(generator=train_examples_iterator,
                                  epochs=nb_epochs,
                                  validation_data=validation_examples_iterator,
                                  callbacks=[epoch_batch_callback],
                                  class_weight=class_weight)

    logger.debug('Done training !')
    train_checkpoint = datetime.now()

    # Saving models.py and history
    serialized_model_file_name = job_folder.split(os.sep)[-2].split('.')[0]  + "_nbepoches_" + str(nb_epochs) + "_nbneg_" + str(nb_neg) + '_model.h5'
    model_file = os.path.join(job_folder, serialized_model_file_name)
    logger.debug(f'Model file name is: {serialized_model_file_name}. Full file is: {model_file}')
    history_file_name = job_folder.split(os.sep)[-2].split('.')[0] + "_nbepoches_" + str(nb_epochs) + "_nbneg_" + str(nb_neg) + '_history.pickle'
    history_file = os.path.join(job_folder, history_file_name)
    logger.debug(f'History file is {history_file}')

    model.save(model_file)
    logger.debug(f"Model saved in {model_file}")
    with open(history_file, "wb") as handle:
        pickle.dump(history.history, handle)

    logger.debug(f"History saved in {model_file}")
    logger.debug(f"Done !")
    logger.debug(f"Training done in      : {train_checkpoint - start_time}")


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

    parser.add_argument('--job_folder', metavar='job_folder',
                        type=str, default='./results/local/',
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
              batch_size=args.batch_size,
              job_folder=args.job_folder)
