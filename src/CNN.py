from models import first_model
from settings import nb_neg_ex_per_pos
from GetCubes import get_cubes
from extraction_data import extract_data
from create_training_examples import create_training_examples
import logging


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler("application.log")
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh.setFormatter(formatter)

    logger.addHandler(fh)


    logger.debug('Calling module extract data.')
    extract_data()

    create_training_examples()

    logger.debug('Creating network model')
    model = first_model()

    # We are taking systems of the first 200 proteins (pos and neg example)

    nb_examples = 200 * (1 + nb_neg_ex_per_pos)
    cubes, ys = get_cubes(nb_examples)

    history = model.fit(cubes, ys)


if __name__ == "__main__":
    main()
