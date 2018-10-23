import numpy as np

from create_examples import create_examples
from settings import EXTRACTED_GIVEN_DATA_TRAIN_FOLDER, TRAINING_EXAMPLES_FOLDER, \
    EXTRACTED_GIVEN_DATA_VALIDATION_FOLDER, VALIDATION_EXAMPLES_FOLDER, TESTING_EXAMPLES_FOLDER, \
    EXTRACTED_PREDICT_DATA_FOLDER, PREDICT_EXAMPLES_FOLDER, EXTRACTED_GIVEN_DATA_TEST_FOLDER

if __name__ == "__main__":
    # To get reproducible generations of examples
    np.random.seed(1337)

    nb_ng_per_pos_to_create = 5

    print(f"Creating training examples with {nb_ng_per_pos_to_create} negatives examples per positive examples")
    create_examples(from_folder=EXTRACTED_GIVEN_DATA_TRAIN_FOLDER,
                    to_folder=TRAINING_EXAMPLES_FOLDER,
                    nb_neg=nb_ng_per_pos_to_create)

    print("Creating validation examples with {nb_ng_per_pos_to_create} negatives examples per positive examples")
    create_examples(from_folder=EXTRACTED_GIVEN_DATA_VALIDATION_FOLDER,
                    to_folder=VALIDATION_EXAMPLES_FOLDER,
                    nb_neg=nb_ng_per_pos_to_create)

    print("Creating testing examples with {nb_ng_per_pos_to_create} negatives examples per positive examples")
    create_examples(from_folder=EXTRACTED_GIVEN_DATA_TEST_FOLDER,
                    to_folder=TESTING_EXAMPLES_FOLDER,
                    nb_neg=nb_ng_per_pos_to_create)

    print("Creating examples for final predictions with {nb_ng_per_pos_to_create} negatives examples per positive "
          "examples")
    create_examples(from_folder=EXTRACTED_PREDICT_DATA_FOLDER,
                    to_folder=PREDICT_EXAMPLES_FOLDER,
                    nb_neg=nb_ng_per_pos_to_create)