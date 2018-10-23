import os, logging
import pickle
import numpy as np
from sklearn import preprocessing
from pandas import DataFrame

from settings import EXTRACTED_GIVEN_DATA_TRAIN_FOLDER, extracted_given_data_validation_folder, NORMALIZED_DATA_TRAIN_FOLDER, \
    NORMALIZED_DATA_TEST_FOLDER, NORMALIZED_DATA_FOLDER, GIVEN_DATA_FOLDER
from pipeline_fixtures import show_progress, load_nparray

logger = logging.getLogger('cnn.Normalizing')
logger.addHandler(logging.NullHandler())


def get_data_scaler(folder: str = EXTRACTED_GIVEN_DATA_TRAIN_FOLDER):
    """
    Return the scaler of data.

    If it doesn't exist, we create it on the go.

    :param folder: the folder to use to create the scaler.
    :return:
    """
    serialized_scaler_file = os.path.join(GIVEN_DATA_FOLDER, "scaler.pickle")
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
    if not (os.path.exists(NORMALIZED_DATA_FOLDER)):
        logger.debug('Creating data folder %s, %s and %s.',
                     NORMALIZED_DATA_FOLDER,
                     NORMALIZED_DATA_TRAIN_FOLDER,
                     NORMALIZED_DATA_TEST_FOLDER)
        os.makedirs(NORMALIZED_DATA_FOLDER)
        os.makedirs(NORMALIZED_DATA_TRAIN_FOLDER)
        os.makedirs(NORMALIZED_DATA_TEST_FOLDER)

    scaler_on_training_data = get_data_scaler(EXTRACTED_GIVEN_DATA_TRAIN_FOLDER)

    normalize_data(scaler_on_training_data, EXTRACTED_GIVEN_DATA_TRAIN_FOLDER, NORMALIZED_DATA_TRAIN_FOLDER)
    normalize_data(scaler_on_training_data, extracted_given_data_validation_folder, NORMALIZED_DATA_TEST_FOLDER)
