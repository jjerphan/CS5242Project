import os
import numpy as np
from keras.optimizers import Adam
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix

# FOLDERS
# Global folder for data and logs
ROOT = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))

JOB_SUBMISSIONS_FOLDER = os.path.join(ROOT, "job_submissions")
LOGS_FOLDER = os.path.join(ROOT, "logs")

for folder in [JOB_SUBMISSIONS_FOLDER, LOGS_FOLDER]:
    if not(os.path.exists(folder)):
        os.makedirs(folder)

# Data given
GIVEN_DATA_FOLDER = os.path.join(ROOT, "training_data")
ORIGINAL_GIVEN_DATA_FOLDER = os.path.join(GIVEN_DATA_FOLDER, "original")
EXTRACTED_GIVEN_DATA_FOLDER = os.path.join(GIVEN_DATA_FOLDER, "extracted")
EXTRACTED_GIVEN_DATA_TRAIN_FOLDER = os.path.join(EXTRACTED_GIVEN_DATA_FOLDER, "train")
EXTRACTED_GIVEN_DATA_VALIDATION_FOLDER = os.path.join(EXTRACTED_GIVEN_DATA_FOLDER, "validation")
EXTRACTED_GIVEN_DATA_TEST_FOLDER = os.path.join(EXTRACTED_GIVEN_DATA_FOLDER, "test")

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

# FEATURE AND REPRESENTATION SETTINGS

# Features used to train:
#  - 3 spatial coordinates : x , y, z (floating values)
#  - 1 features for one hot encoding of atom types (is_hydrophobic)
#  - 1 features for one hot encoding of molecules types (is_from_protein)
FEATURES_NAMES = ["x", "y", "z", "is_hydrophobic", "is_polar", "is_from_protein", "is_from_ligand"]
# Obtained with values_range on the complete original dataset
HYDROPHOBIC_TYPES = {'h', 'C'}
POLAR_TYPES = {'p', 'P', 'O', 'TE', 'F', 'N', 'AS', 'O1-', 'MO',
               'B', 'BR', 'SB', 'RU', 'SE', 'HG', 'CL',
               'S', 'FE', 'ZN', 'CU', 'SI', 'V', 'I', 'N+1',
               'N1+', 'CO', 'W'}

NB_FEATURES = len(FEATURES_NAMES)
NB_CHANNELS = NB_FEATURES - 3  # coordinates are not used as features then

# A mapping from names to their actual index
INDICES_FEATURES = dict(zip(FEATURES_NAMES, list(range(NB_FEATURES))))

# To scale protein-ligands system in a cube of shape (LENGTH_CUBE_SIDE,LENGTH_CUBE_SIDE,LENGTH_CUBE_SIDE)
LENGTH_CUBE_SIDE = 20
DEFAULT_CUBE_RES = 3.0
SHAPE_CUBE = (LENGTH_CUBE_SIDE, LENGTH_CUBE_SIDE, LENGTH_CUBE_SIDE, NB_CHANNELS)

# PREPROCESSING SETTINGS

# We have 3000 positives pairs of ligands: we keep 80% for training, 10% for evaluation and 10% for testing
PERCENT_TRAIN = 0.8
PERCENT_TEST = 0.1

# The maximum number of negative example to create per positive example to train
# Used for creating training examples
MAX_NB_NEG_PER_POS = 20

# Pre-processing settings
NB_WORKERS = 6
EXTRACTED_PROTEIN_SUFFIX = "_pro_cg.csv"
EXTRACTED_LIGAND_SUFFIX = "_lig_cg.csv"
FLOAT_TYPE = np.float32
FORMATTER = "%.16f"
COMMENT_DELIMITER = "#"

# JOBS SETTINGS

# The environment to use for jobs
JOBS_ENV = "CS5242_gpu"

# Training settings
TRAINING_LOGFILE = f"train_cnn.log"
RESULTS_FOLDER = os.path.join(ROOT, "results")
JOB_FOLDER_DEFAULT = os.path.join(RESULTS_FOLDER, "local")

PARAMETERS_FILE_NAME_SUFFIX = "parameters.txt"
SERIALIZED_MODEL_FILE_NAME_SUFFIX = "model.h5"
HISTORY_FILE_NAME_SUFFIX = "history.pickle"
NB_EPOCHS_DEFAULT = 1
BATCH_SIZE_DEFAULT = 32
N_GPU_DEFAULT = 1
OPTIMIZER_DEFAULT = Adam()

# The number of negative example to use per positive example to train
NB_NEG_EX_PER_POS = 10

# A constant to reduce the weight of the positive class
WEIGHT_POS_CLASS = 2

# Evaluation settings
METRICS_FOR_EVALUATION = [accuracy_score, precision_score, recall_score, f1_score, confusion_matrix]
EVALUATION_LOGS_FOLDER = os.path.join(RESULTS_FOLDER, "evaluation")
EVALUATION_CSV_FILE = os.path.join(EVALUATION_LOGS_FOLDER, "evaluation_results.csv")


