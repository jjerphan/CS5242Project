from models import first_model
from settings import nb_neg_ex_per_pos, model_weights, whole_model
from GetCubes import get_cubes
from extraction_data import extract_data
from create_training_examples import create_training_examples
import logging


def main():
    """
    1. Extract data from the given pdb file.
    2. Create training examples by mixing protein and ligand bindings, both positive and negative bindings are created.
    3.
    :return:
    """
    logger = logging.getLogger('cnn')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("application.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


    logger.debug('Calling module extract data and training examples.')
    # extract_data()

    # create_training_examples()

    logger.debug('Creating network model.')
    model = first_model()

    # We are taking systems of the first 200 proteins (pos and neg example)
    nb_examples = 200 * (1 + nb_neg_ex_per_pos)
    logger.debug('Creating cubes.')
    cubes, ys = get_cubes(nb_examples)

    logger.debug('Training the model.')
    history = model.fit(cubes, ys)

    logger.debug(('Saving model.'))
    model.save(whole_model)


if __name__ == "__main__":
    main()
