import os
import pickle
import csv
from keras.models import load_model
from settings import original_predict_folder, extracted_predict_folder, predict_examples_folder, results_folder

from extraction_data import extract_data
from create_examples import create_examples
from PredictGenerator import PredictGenerator

def predict(model, preprocessing=False):
    # Preprocessing - 1. Extract data from original pdb file and create as molecues.
    #                 2. Mix each protein with all ligand bindings for prediction
    if preprocessing:
        extract_data(original_predict_folder, False)
        create_examples(extracted_predict_folder, predict_examples_folder)

    # Load pre-trained good model
    my_model = load_model(model)

    matching = {}

    predict_examples_generator = PredictGenerator(predict_examples_folder)
    for pro, lig, cube in predict_examples_generator:
        y_predict = my_model.predict(cube)
        try:
            matching[pro]
        except KeyError:
            matching[pro] = []

        matching[pro].append((y_predict[0][0], lig))

    with open(os.path.join(results_folder, 'matching.pkl'), 'wb') as f:
        pickle.dump(matching, f)
    
    matching_list = []
    with open(os.path.join(results_folder, 'result.csv'), 'w') as f:
        csvwriter = csv.writer(f)
        for pro, value in sorted(matching.items()):
            top_10 = sorted(value, reverse=True)[:10]
            top_10_ligents = list(map(lambda x: x[1], top_10))
            row = [pro + ", " + ", ".join(top_10_ligents)]
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

if __name__ == '__main__':
    predict('./results/2018-10-15_15:28:21.109066+08:00/model.h5')
