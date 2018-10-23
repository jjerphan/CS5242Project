import argparse
import os
import pickle
import csv
import logging
import keras

from collections import defaultdict

from keras.models import load_model

from discretization import AbsoluteCubeRepresentation, RelativeCubeRepresentation
from pipeline_fixtures import get_parameters_dict
from settings import TESTING_EXAMPLES_FOLDER, LENGTH_CUBE_SIDE
from predict_generator import PredictGenerator
from settings import PREDICT_EXAMPLES_FOLDER, RESULTS_FOLDER
from train_cnn import f1


def perform_matching(predictions: dict, nb_top_ligands: int):
    """
    Perform a simple matching using predictions: for each protein,
    the nb_top_ligands best ligands are chosen.

    Prediction is a dictionary of the form:

        {
            pro_id : [(affinity, lig_id) ...],
            ...
        }


    :param predictions : the dictionary to use.
    :param nb_top_ligands: the number of ligand to consider in the list
    :return: a list of list of the form [[pro, lig1_id, lig2_id, … lignb_top_ligands_id]]
    """
    matching_list = []
    for pro, ligands_scores in sorted(predictions.items()):
        top_scores = sorted(ligands_scores, reverse=True)[:nb_top_ligands]
        top_ligands = list(map(lambda x: x[1], top_scores))
        protein_with_best_ligands = [pro, *top_ligands]
        matching_list.append(protein_with_best_ligands)

    return matching_list


def calculate_success_rate(matching_list: list):
    """
    Compute the final success rate using a matching list

    :param matching_list:
    :return: a number in [0,1] representing the final success rate
    """
    success_count = 0
    failure_count = 0
    for item in matching_list:
        protein_indice = item[0]
        ligand_indices = item[1:]
        if protein_indice in ligand_indices:
            success_count += 1
        else:
            failure_count += 1
    return success_count / (success_count + failure_count)


def predict(serialized_model_path, evaluation=True):
    """
    Predict the results with a model.

    :param serialized_model_path: the file that contains the serialized model
    :param evaluation: if true, it evaluates the performances of the model instead of prediction
    :return:
    """

    nb_top_ligands = 10

    # Formatting Fixtures
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    id = serialized_model_path.split(os.sep)[-2]
    job_folder = os.path.join(RESULTS_FOLDER, id)
    if not (os.path.exists(RESULTS_FOLDER)):
        print(f"The {RESULTS_FOLDER} does not exist. Creating it.")
        os.makedirs(RESULTS_FOLDER)

    fh = logging.FileHandler(os.path.join(job_folder, f'{"" if evaluation else "final_"}prediction.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    predictions_file_name = os.path.join(job_folder, f"{id}_matching.pkl")
    result_file_name = os.path.join(job_folder, f"{id}_result.txt")

    # Choose the corrected folder because we can evaluate (to test the performance of a model)
    # or because we can predict
    if evaluation:
        predict_folder = TESTING_EXAMPLES_FOLDER
    else:
        predict_folder = PREDICT_EXAMPLES_FOLDER
    logger.debug(f'Using example folder: {predict_folder}.')

    model = load_model(serialized_model_path, custom_objects={'f1': f1})
    logger.debug(f'Model loaded. Summary: ')
    keras.utils.print_summary(model, print_fn=logger.debug)

    parameters = get_parameters_dict(job_folder=job_folder)

    cube_representation = AbsoluteCubeRepresentation(length_cube_side=LENGTH_CUBE_SIDE) \
        if parameters["representation"] == AbsoluteCubeRepresentation.name \
        else RelativeCubeRepresentation(length_cube_side=LENGTH_CUBE_SIDE)

    # Getting predictions
    predictions = defaultdict(list)

    predict_examples_generator = PredictGenerator(predict_folder,
                                                  representation=cube_representation)
    for pro, lig, cube in predict_examples_generator:
        y_predict = model.predict(cube)
        predictions[pro].append((y_predict[0][0], lig))

    # Saving predictions
    with open(predictions_file_name, 'wb') as f:
        pickle.dump(predictions, f)
        logger.debug(f'Matching pickle file saved {predictions_file_name}')

    # Getting the matching
    matching_list = perform_matching(predictions, nb_top_ligands)

    with open(os.path.join(result_file_name), 'w') as f:
        csv_writer = csv.writer(f)
        headers = ["pro_id", *[f"lig{i}_id" for i in range(1, nb_top_ligands+1)]]
        csv_writer.writerow(headers)
        csv_writer.writerows(matching_list)
        logger.debug(f'Result file saved {result_file_name}')

    success_rate = calculate_success_rate(matching_list)
    logger.debug(f'Success rate for model {id} is : {success_rate}')


if __name__ == "__main__":
    # Parsing sysargv arguments
    parser = argparse.ArgumentParser(description='Evaluate a model using a serialized version of it.')

    parser.add_argument('--model_path', metavar='model_path',
                        type=str, required=True,
                        help=f'where the serialized file of the model (.h5) is.')

    parser.add_argument('--evaluation', metavar='evaluation',
                        type=bool, default=True,
                        help='if true: action on test data from training set')

    args = parser.parse_args()

    print("Argument parsed : ", args)

    predict(serialized_model_path=args.model_path,
            evaluation=args.evaluation)
