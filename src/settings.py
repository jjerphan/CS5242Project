import os
import numpy as np

# Folders
# Global folder for data and logs
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, auc

absolute_path = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))

job_submissions_folder = os.path.join(absolute_path, "job_submissions")

# Data given (not modified)
data_folder = os.path.join(absolute_path, "training_data")
original_data_folder = os.path.join(data_folder, "original")

# First extraction of data
extracted_data_folder = os.path.join(data_folder, "extracted")
extracted_data_train_folder = os.path.join(extracted_data_folder, "train")
extracted_data_test_folder = os.path.join(extracted_data_folder, "test")
normalized_data_folder = os.path.join(data_folder, "normalized")
normalized_data_train_folder = os.path.join(normalized_data_folder, "train")
normalized_data_test_folder = os.path.join(normalized_data_folder, "test")

predict_folder = os.path.join(absolute_path, "predict_data")
original_predict_folder = os.path.join(predict_folder, "original")
extracted_predict_folder = os.path.join(predict_folder, "extracted")

# Conversion to examples
training_examples_folder = os.path.join(data_folder, "training_examples")
testing_examples_folder = os.path.join(data_folder, "testing_examples")
predict_examples_folder = os.path.join(predict_folder, "predict_examples")

# Suffixes for extracted data files
extracted_protein_suffix = "_pro_cg.csv"
extracted_ligand_suffix = "_lig_cg.csv"

# Results
training_logfile = f"train_cnn.log"
results_folder = os.path.join(absolute_path, "results")
parameters_file_name = "parameters.txt"
serialized_model_file_name = "model.h5"
history_file_name = "history.pickle"

# Some settings for number and persisting tensors
float_type = np.float32
formatter = "%.16f"
comment_delimiter = "#"

# Features used to train:
#  - 3 spatial coordinates : x , y, z (floating values)
#  - 1 features for one hot encoding of atom types (is_hydrophobic)
#  - 1 features for one hot encoding of molecules types (is_from_protein)

features_names = ["x", "y", "z", "is_hydrophobic", "is_from_protein"]
nb_features = len(features_names)
nb_channels = nb_features - 3  # coordinates are not used as features
indices_features = dict(zip(features_names, list(range(nb_features))))

# We have 3000 positives pairs of ligands
percent_train = 0.8
percent_test = 0.1
n_training_examples = 2700

# All the other will be used to construct examples on the go

# We augment the number of examples
nb_neg_ex_per_pos = 10

# To scale protein-ligands system in a cube of shape (resolution_cube,resolution_cube,resolution_cube)
length_cube_side = 20
shape_cube = (length_cube_side, length_cube_side, length_cube_side, 2)

# Obtained with values_range on the complete original dataset : to be rerun again
hydrophobic_types = {"C"}
polar_types = {'P', 'O', 'TE', 'F', 'N', 'AS', 'O1-', 'MO',
               'B', 'BR', 'SB', 'RU', 'SE', 'HG', 'CL',
               'S', 'FE', 'ZN', 'CU', 'SI', 'V', 'I', 'N+1',
               'N1+', 'CO', 'W', }

# Training parameters
nb_epochs_default = 1
batch_size_default = 32
n_gpu_default = 1
optimizer_default = "rmsprop"

# Evaluation parameters
metrics_for_evaluation = [accuracy_score, precision_score, recall_score, f1_score]

# Pre-processing settings
nb_workers = 6
name_env = "CS5242_gpu"


