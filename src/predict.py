import argparse
import os
import pickle
import csv
import logging
import keras

from collections import defaultdict

from keras.models import load_model
from settings import predict_examples_folder, results_folder, nb_neg_ex_per_pos, testing_examples_folder
from predict_generator import PredictGenerator
from settings import predict_examples_folder, results_folder


def predict(serialized_model_path, nb_neg, max_examples, verbose=1, evaluation=True):
    """
    :param serialized_model_path: where the serialized_model is
    :param max_examples: the maximum number of examples to use
    :param verbose: to have verbose outputs
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(results_folder, 'prediction.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    model_name = serialized_model_path.split('/')[-1]
    matching_file_name = os.path.join(results_folder, model_name.replace("model.h5", "matching.pkl"))
    result_file_name = os.path.join(results_folder, model_name.replace("model.h5", "result.txt"))

    if evaluation:
        predict_folder = testing_examples_folder
    else:
        predict_folder = predict_examples_folder
    logger.debug(f'Using example folder: {predict_folder}.')

    # Load pre-trained good model
    my_model = load_model(serialized_model_path)
    logger.debug(f'Model loaded. Summary: ')
    keras.utils.print_summary(my_model, print_fn=logger.debug)

    matching = defaultdict(list)

    predict_examples_generator = PredictGenerator(predict_folder)
    for pro, lig, cube in predict_examples_generator:
        y_predict = my_model.predict(cube)

        matching[pro].append((y_predict[0][0], lig))

    with open(matching_file_name, 'wb') as f:
        pickle.dump(matching, f)
        logger.debug(f'Matching pickle file saved {matching_file_name}')
    
    matching_list = []
    with open(result_file_name, 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow('pro_id  lig1_id lig2_id lig3_id lig4_id lig5_id lig6_id lig7_id lig8_id lig9_id lig10_id')
        for pro, value in sorted(matching.items()):
            top_10 = sorted(value, reverse=True)[:10]
            top_10_ligands = list(map(lambda x: x[1], top_10))
            row = [pro + "  " + "   ".join(top_10_ligands)]
            matching_list.append(row)
            csvwriter.writerow(row)
        logger.debug(f'Result file saved {result_file_name}')

    def cal_success_rate(matching_list=matching_list):
        success_count = 0
        failure_count = 0
        for item in matching_list:
            protein_indice = item[0]
            ligend_indices = item[1:]
            if protein_indice in ligend_indices:
                success_count += 1
            else:
                failure_count += 1
        return success_count / (success_count + failure_count)

    success_rate = cal_success_rate()
    logger.debug(f'Success rate for model {model_name} is : {success_rate}')

    
if __name__ == "__main__":
    # Parsing sysargv arguments
    parser = argparse.ArgumentParser(description='Evaluate a model using a serialized version of it.')

    parser.add_argument('--model_path', metavar='model_path',
                        type=str, required=True,
                        help=f'where the serialized file of the model (.h5) is.')

    parser.add_argument('--max_examples', metavar='max_examples',
                        type=int, default=None,
                        help='the number of total examples to use in total')

    parser.add_argument('--verbose', metavar='verbose',
                        type=int, default=True,
                        help='the number of total examples to use in total')

    parser.add_argument('--evaluation', metavar='evaluation',
                        type=bool, default=True,
                        help='if true: action on test data from training set')

    args = parser.parse_args()

    print("Argument parsed : ", args)

    assert (args.nb_neg > 0)
    assert (args.nb_neg > 0)

    predict(serialized_model_path=args.model_path,
            nb_neg=args.nb_neg,
            max_examples=args.max_examples,
            verbose=args.verbose,
            evaluation=args.evaluation)