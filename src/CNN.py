from models import first_model
from settings import nb_neg_ex_per_pos
from GetCubes import get_cubes
from extraction_data import extract_data
from create_training_examples import create_training_examples


def main():
    extract_data()

    create_training_examples()

    model = first_model()

    # We are taking systems of the first 200 proteins (pos and neg example)
    nb_examples = 200 * (1 + nb_neg_ex_per_pos)
    cubes, ys = get_cubes(nb_examples)

    history = model.fit(cubes, ys)


if __name__ == "__main__":
    main()