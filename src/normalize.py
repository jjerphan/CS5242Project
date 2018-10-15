import os, logging
import pickle
import numpy as np
from sklearn import preprocessing
from pandas import DataFrame

from settings import extracted_data_train_folder, extracted_data_test_folder, normalized_data_train_folder, \
    normalized_data_test_folder, normalized_data_folder, data_folder
from pipeline_fixtures import show_progress
from discretization import load_nparray

logger = logging.getLogger('cnn.Normalizing')
logger.addHandler(logging.NullHandler())


def get_data_scaler(folder: str = extracted_data_train_folder):
    """
    Return the scaler of data.

    If it doesn't exist, we create it on the go.

    :param folder: the folder to use to create the scaler.
    :return:
    """
    serialized_scaler_file = os.path.join(data_folder, "scaler.pickle")
    try:
        scaler = pickle.load(open(serialized_scaler_file, "rb"))
    except:
        logger.debug("Creating the scaler")
        data_for_scaling = DataFrame()
        for file in show_progress(os.listdir(folder)):
            data = load_nparray(os.path.join(folder, file))
            data_for_scaling.append(DataFrame(data[:, 0:3]))

        scaler = preprocessing.StandardScaler()
        scaler = scaler.fit(data_for_scaling)
        pickle.dump(scaler, open(serialized_scaler_file, "wb"))

    return scaler


def normalize_data(scaler, extracted_data_folder, normalized_data_folder):
    """
    Normalize the data from a folder `extracted_data_folder` and
    save it in the folder `normalized_data_folder`.

    :param scaler:
    :param extracted_data_folder:
    :param normalized_data_folder:
    :return:
    """
    logger.debug('Creating scaler.')
    logger.debug('Generating normalized test data.')

    files = sorted(os.listdir(extracted_data_folder))
    for file in show_progress(files):
        data = load_nparray(os.path.join(extracted_data_folder, file))
        new_data = np.append(scaler.transform(data[:, :3]), data[:, 3:], axis=1)
        file_name = os.path.join(normalized_data_folder, file)
        np.savetxt(file_name, new_data)


if __name__ == "__main__":
    if not (os.path.exists(normalized_data_folder)):
        logger.debug('Creating data folder %s, %s and %s.',
                     normalized_data_folder,
                     normalized_data_train_folder,
                     normalized_data_test_folder)
        os.makedirs(normalized_data_folder)
        os.makedirs(normalized_data_train_folder)
        os.makedirs(normalized_data_test_folder)

    scaler_on_training_data = get_data_scaler(extracted_data_train_folder)

    normalize_data(scaler_on_training_data, extracted_data_train_folder, normalized_data_train_folder)
    normalize_data(scaler_on_training_data, extracted_data_test_folder, normalized_data_test_folder)
