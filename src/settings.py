import os
import numpy as np
from keras.optimizers import Adam
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix

# Folders
# Global folder for data and logs
ROOT = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))

JOB_SUBMISSIONS_FOLDER = os.path.join(ROOT, "job_submissions")

# Data given
GIVEN_DATA_FOLDER = os.path.join(ROOT, "training_data")
ORIGINAL_GIVEN_DATA_FOLDER = os.path.join(GIVEN_DATA_FOLDER, "original")
EXTRACTED_GIVEN_DATA_FOLDER = os.path.join(GIVEN_DATA_FOLDER, "extracted")
EXTRACTED_GIVEN_DATA_TRAIN_FOLDER = os.path.join(EXTRACTED_GIVEN_DATA_FOLDER, "train")
extracted_given_data_validation_folder = os.path.join(EXTRACTED_GIVEN_DATA_FOLDER, "validation")
extracted_given_data_test_folder = os.path.join(EXTRACTED_GIVEN_DATA_FOLDER, "test")

# Conversion to examples
TRAINING_EXAMPLES_FOLDER = os.path.join(GIVEN_DATA_FOLDER, "training_examples")
VALIDATION_EXAMPLES_FOLDER = os.path.join(GIVEN_DATA_FOLDER, "validation_examples")
TESTING_EXAMPLES_FOLDER = os.path.join(GIVEN_DATA_FOLDER, "testing_examples")

# Not used for now
NORMALIZED_DATA_FOLDER = os.path.join(GIVEN_DATA_FOLDER, "normalized")
NORMALIZED_DATA_TRAIN_FOLDER = os.path.join(NORMALIZED_DATA_FOLDER, "train")
NORMALIZED_DATA_VALIDATE_FOLDER = os.path.join(NORMALIZED_DATA_FOLDER, "validate")
NORMALIZED_DATA_TEST_FOLDER = os.path.join(NORMALIZED_DATA_FOLDER, "test")

# Data to predict final results
PREDICT_DATA_FOLDER = os.path.join(ROOT, "testing_data_release")
ORIGINAL_PREDICT_DATA_FOLDER = os.path.join(PREDICT_DATA_FOLDER, "testing_data")
EXTRACTED_PREDICT_DATA_FOLDER = os.path.join(PREDICT_DATA_FOLDER, "extracted")

PREDICT_EXAMPLES_FOLDER = os.path.join(PREDICT_DATA_FOLDER, "predict_examples")

# Suffixes for extracted data files
EXTRACTED_PROTEIN_SUFFIX = "_pro_cg.csv"
EXTRACTED_LIGAND_SUFFIX = "_lig_cg.csv"

# Results
TRAINING_LOGFILE = f"train_cnn.log"
RESULTS_FOLDER = os.path.join(ROOT, "results")
PARAMETERS_FILE_NAME = "parameters.txt"
SERIALIZED_MODEL_FILE_NAME = "model.h5"

# Some settings for number and persisting tensors
FLOAT_TYPE = np.float32
FORMATTER = "%.16f"
COMMENT_DELIMITER = "#"

# Features used to train:
#  - 3 spatial coordinates : x , y, z (floating values)
#  - 1 features for one hot encoding of atom types (is_hydrophobic)
#  - 1 features for one hot encoding of molecules types (is_from_protein)

FEATURES_NAMES = ["x", "y", "z", "is_hydrophobic", "is_from_protein"]
NB_FEATURES = len(FEATURES_NAMES)
NB_CHANNELS = NB_FEATURES - 3  # coordinates are not used as features
INDICES_FEATURES = dict(zip(FEATURES_NAMES, list(range(NB_FEATURES))))

# We have 3000 positives pairs of ligands
PERCENT_TRAIN = 0.8
PERCENT_TEST = 0.1
N_TRAINING_EXAMPLES = 2400

# All the other will be used to construct examples on the go

# The maximum number of negative example to create per positive example to train
# Used for creating training examples
MAX_NB_NEG_PER_POS = 20

# The number of negative example to use per positive example to train
NB_NEG_EX_PER_POS = 10

# To scale protein-ligands system in a cube of shape (resolution_cube,resolution_cube,resolution_cube)
LENGTH_CUBE_SIDE = 20
SHAPE_CUBE = (LENGTH_CUBE_SIDE, LENGTH_CUBE_SIDE, LENGTH_CUBE_SIDE, 2)

# Obtained with values_range on the complete original dataset
HYDROPHOBIC_TYPES = {'h', 'C'}
POLAR_TYPES = {'p', 'P', 'O', 'TE', 'F', 'N', 'AS', 'O1-', 'MO',
               'B', 'BR', 'SB', 'RU', 'SE', 'HG', 'CL',
               'S', 'FE', 'ZN', 'CU', 'SI', 'V', 'I', 'N+1',
               'N1+', 'CO', 'W'}

# Training parameters
NB_EPOCHS_DEFAULT = 1
BATCH_SIZE_DEFAULT = 32
N_GPU_DEFAULT = 1
OPTIMIZER_DEFAULT = Adam()

SERIALIZED_MODEL_FILE_NAME_END = "model.h5"
HISTORY_FILE_NAME_END = "history.pickle"

# Evaluation parameters
METRICS_FOR_EVALUATION = [accuracy_score, precision_score, recall_score, f1_score, confusion_matrix]

# Pre-processing settings
NB_WORKERS = 6
NAME_ENV = "CS5242_gpu"


