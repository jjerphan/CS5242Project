import argparse
import os
import pickle
import csv

from collections import defaultdict
from keras.models import load_model
from settings import original_predict_folder, extracted_predict_folder, predict_examples_folder, results_folder, \
    nb_neg_ex_per_pos

from extraction_data import extract_data
from create_examples import create_examples
from predict_generator import PredictGenerator


def predict(serialized_model_path, nb_neg, max_examples, verbose=1, preprocess=False):
    """
    Preprocessing - 1. Extract data from original pdb file and create as molecues.
                    2. Mix each protein with all ligand bindings for prediction

    :param serialized_model_path: where the serialized_model is
    :param nb_neg: the number of negative examples to use per positive examples
    :param max_examples: the maximum number of examples to use
    :param verbose: to have verbose outputs
    :param preprocess: should we do some pre-processing
    :return:
    """
    if preprocess:
        extract_data(original_predict_folder, split_validation_testing=False)
        create_examples(extracted_predict_folder, predict_examples_folder)

    # Load pre-trained good model
    my_model = load_model(serialized_model_path)

    matching = defaultdict(list)

    predict_examples_generator = PredictGenerator(predict_examples_folder)
    for pro, lig, cube in predict_examples_generator:
        y_predict = my_model.predict(cube)

        matching[pro].append((y_predict[0][0], lig))

    with open(os.path.join(results_folder, 'matching.pkl'), 'wb') as f:
        pickle.dump(matching, f)
    
    matching_list = []
    with open(os.path.join(results_folder, 'result.csv'), 'w') as f:
        csvwriter = csv.writer(f)
        for pro, value in sorted(matching.items()):
            top_10 = sorted(value, reverse=True)[:10]
            top_10_ligands = list(map(lambda x: x[1], top_10))
            row = [pro + ", " + ", ".join(top_10_ligands)]
            matching_list.append(row)
            csvwriter.writerow(row)

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

    print(cal_success_rate())

    
if __name__ == "__main__":
    # Parsing sysargv arguments
    parser = argparse.ArgumentParser(description='Evaluate a model using a serialized version of it.')

    parser.add_argument('--model_path', metavar='model_path',
                        type=str, required=True,
                        help=f'where the serialized file of the model (.h5) is.')

    parser.add_argument('--nb_neg', metavar='nb_neg',
                        type=int, default=nb_neg_ex_per_pos,
                        help='the number of negatives examples to use per positive example')

    parser.add_argument('--max_examples', metavar='max_examples',
                        type=int, default=None,
                        help='the number of total examples to use in total')

    parser.add_argument('--verbose', metavar='verbose',
                        type=int, default=True,
                        help='the number of total examples to use in total')

    parser.add_argument('--preprocess', metavar='preprocess',
                        type=int, default=True,
                        help='if !=0 triggers the pre-processing of the data')

    args = parser.parse_args()

    print("Argument parsed : ", args)

    assert (args.nb_neg > 0)
    assert (args.nb_neg > 0)

    predict(serialized_model_path=args.model_path,
            nb_neg=args.nb_neg,
            max_examples=args.max_examples,
            verbose=args.verbose,
            preprocess=args.preprocess)

