import os

import progressbar
import numpy as np

# Folders
# Global folder for data
data_folder = os.path.join("..", "training_data")
# Data given (not modified)
original_data = os.path.join(data_folder, "original")
# First extraction of data
extracted_data = os.path.join(data_folder, "extracted")
# Conversion to examples
examples_data = os.path.join(data_folder, "examples")

# Suffixes for extracted data files
protein_suffix = "_pro_cg.csv"
ligand_suffix = "_lig_cg.csv"

# Some settings for number and persisting tensors
float_type = np.float32
formatter = "%.16f"
comment_delimiter = "#"

# Categorical features encoding
# Encoded values for atom
hydrophobic_value = 1
polar_value = -1
# Encoded values for molecules
protein_value = 1
ligand_value = -1

# To scale protein-ligands system in a cube of shape (resolution_cube,resolution_cube,resolution_cube)
resolution_cube = 30

# Obtained with values_range on the complete original dataset
hydrophobic_types = {"C"}
polar_types = {'P', 'O', 'TE', 'F', 'N', 'AS', 'O1-', 'MO',
               'B', 'BR', 'SB', 'RU', 'SE', 'HG', 'CL',
               'S', 'FE', 'ZN', 'CU', 'SI', 'V', 'I', 'N+1',
               'N1+', 'CO', 'W', }

x_min = -244.401
x_max = 310.935

y_min = -186.407
y_max = 432.956

z_mix = -177.028
z_max = 432.956
##

# Misc.
widgets_progressbar = [
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
]