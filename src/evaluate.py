import pandas as pd
import numpy as np
import seaborn as sn
from keras.models import load_model

from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, confusion_matrix

from matplotlib import pyplot as plt

from pipeline_fixtures import ExamplesIterator
from settings import testing_examples_folder, nb_workers


def evaluate_model(serialized_model):
    model = load_model(serialized_model)

    test_examples_iterator = ExamplesIterator(examples_folder=testing_examples_folder)
    model.evaluate_generator(test_examples_iterator, workers=nb_workers)

    # or
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
    serialized_model = ""
    evaluate_model(serialized_model)
