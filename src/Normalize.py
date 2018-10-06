from sklearn import preprocessing
from settings import extracted_data_train_folder, extracted_data_test_folder, normalized_data_train_folder, normalized_data_test_folder, normalized_data_folder, progress
import os, logging
from pandas import DataFrame
from discretization import load_nparray
import numpy as np


logger = logging.getLogger('cnn.Normalizing')
logger.addHandler(logging.NullHandler())

def read_training_data():
    training_files = os.listdir(extracted_data_train_folder)
    training_x_y_z = DataFrame()
    for file in progress(training_files):
        data = load_nparray(os.path.join(extracted_data_train_folder, file))
        training_x_y_z = training_x_y_z.append(DataFrame(data[:, 0:3]))
    return training_x_y_z


def scaling():
    data = read_training_data()
    scaler = preprocessing.StandardScaler()
    scaler = scaler.fit(data)
    return scaler


def normalized_data():
    if not(os.path.exists(normalized_data_folder)):
        logger.debug('Creating data folder %s, %s and %s.', normalized_data_folder, normalized_data_train_folder, normalized_data_test_folder)
        os.makedirs(normalized_data_folder)
        os.makedirs(normalized_data_train_folder)
        os.makedirs(normalized_data_test_folder)

    logger.debug('Creating scaler.')
    scaler = scaling()
    training_files = sorted(os.listdir(extracted_data_train_folder))

    logger.debug('Generating normalized training data.')
    for file in progress(training_files):
        data = load_nparray(os.path.join(extracted_data_train_folder, file))
        new_data = np.append(scaler.transform(data[:, :3]), data[:, 3:], axis=1)
        file_name = os.path.join(normalized_data_train_folder, file)
        np.savetxt(file_name, new_data)

    logger.debug('Generating normalized test data.')
    test_files = sorted(os.listdir(extracted_data_test_folder))
    for file in progress(test_files):
        data = load_nparray(os.path.join(extracted_data_test_folder, file))
        new_data = np.append(scaler.transform(data[:, :3]), data[:, 3:], axis=1)
        file_name = os.path.join(normalized_data_train_folder, file)
        np.savetxt(file_name, new_data)


if __name__ == "__main__":
    normalized_data()