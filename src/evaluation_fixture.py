import pandas as pd
import numpy as np
import seaborn as sn

from pipeline_fixtures import examples_iterator
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, confusion_matrix
from settings import progress, extracted_data_test_folder

from matplotlib import pyplot as plt

# Fixtures got from the note book : to be sorted

def plot_history(history):
    plt.plot(history.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def get_cubes_testing(nb_examples=128):
    """
    Return the first nb_examples cubes for testing with their ys.
    :param nb_examples:
    :return: list of cubes and list of their ys
    """
    counter = 1
    cubes = []
    ys = []
    protein_ids = []
    ligand_ids = []
    for cube, protein_id, ligand_id in progress(examples_iterator(extracted_data_test_folder)):
        counter +=1
        y = 1 * (protein_id == ligand_id)
        protein_ids.append(protein_id)
        ligand_ids.append(ligand_id)
        cubes.append(cube)
        ys.append(y)
        if counter > nb_examples:
            break

    # Conversion to np.ndarrays with the first axes used for examples
    cubes = np.array(cubes)
    ys = np.array(ys)
    protein_ids = np.array(protein_ids)
    ligand_ids = np.array(ligand_ids)
    assert(ys.shape[0] == nb_examples)
    assert(cubes.shape[0] == nb_examples)

    return cubes, ys, protein_ids, ligand_ids


if __name__ == "__main__":
    cubes_test, ys, protein_ids, ligand_ids = get_cubes_testing()


    y_preds = np.array(list(map(lambda x: x[0], preds)))

    # Rounding the prediction : using the second one
    y_rounded = np.array([1 if y!= 0 else 0 for y in y_preds])


    conf_matrix = confusion_matrix(ys, y_rounded)

    df_cm = pd.DataFrame(conf_matrix, index = ["Neg", "Pos"],
                      columns = ["Neg", "Pos"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)


    print(f"F1 score : {f1_score(ys,y_rounded)}")
    print(f"Accuracy score : {accuracy_score(ys,y_rounded)}")
    print(f"Precision score : {precision_score(ys,y_rounded)}")
    print(f"Recall score : {recall_score(ys,y_rounded)}")

    # Counting non negative predictions
    len(list(filter(lambda x: x!= 0, y_preds)))