import pandas as pd
import numpy as np
import seaborn as sn
from keras.models import load_model

from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, confusion_matrix

from matplotlib import pyplot as plt

from create_job_sub import ModelInspector
from pipeline_fixtures import ExamplesIterator
from settings import testing_examples_folder, nb_workers, models_folders


def evaluate_model(serialized_model):
    """

    :param serialized_model:
    :return:
    """
    model = load_model(serialized_model)

    test_examples_iterator = ExamplesIterator(examples_folder=testing_examples_folder,
                                              shuffle_after_completion=False)
    model.evaluate_generator(test_examples_iterator, workers=nb_workers)

    ys = test_examples_iterator.get_labels()

    y_preds = model.predict_generator(test_examples_iterator)

    # Rounding the prediction : using the second one
    y_rounded = np.array([1 if y != 0 else 0 for y in y_preds])

    conf_matrix = confusion_matrix(ys, y_rounded)

    df_cm = pd.DataFrame(conf_matrix, index=["Neg", "Pos"],
                         columns=["Neg", "Pos"])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)

    print(f"F1 score : {f1_score(ys,y_rounded)}")
    print(f"Accuracy score : {accuracy_score(ys,y_rounded)}")
    print(f"Precision score : {precision_score(ys,y_rounded)}")
    print(f"Recall score : {recall_score(ys,y_rounded)}")

    # Counting non negative predictions
    len(list(filter(lambda x: x != 0, y_preds)))


if __name__ == "__main__":

    model_inspector = ModelInspector(models_folders=models_folders)
    model_index = -1
    while model_index not in range(len(model_inspector)):
        print("Choose the model to evaluate")
        for index, (folder, set_parameters, chosen_model, history) in enumerate(model_inspector):
            print(f"#{index} Name : {set_parameters['name']} (from {folder})")
            for key, value in set_parameters.items():
                print(f" - {key}: {value}")

        model_index = int(input("Your choice : # "))

    _, _, chosen_model, _ = model_inspector[model_index]
    print(chosen_model)
    evaluate_model(chosen_model)
