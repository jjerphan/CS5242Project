import datetime

import keras
import numpy as np
import progressbar

from settings import FLOAT_TYPE, COMMENT_DELIMITER

widgets_progressbar = [
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar("░", fill="⋅"),
    ' (', progressbar.ETA(), ') ',
]


def is_positive(name):
    """
    name is of the form "xxxx_yyyy[.csv]"

    'xxxx' corresponds to the protein.
    'yyyy' corresponds to the ligands

    Return xxxx == yyyy (both molecules bind together)

    :param name: the name of a file of the form "xxxx_yyyy[.csv]"
    :return:
    """
    systems = name.replace(".csv", "").split("_")
    return systems[0] == systems[1]


def is_negative(name):
    """
    name is of the form "xxxx_yyyy[.csv]"

    'xxxx' corresponds to the protein.
    'yyyy' corresponds to the ligands

    Return xxxx != yyyy (both molecules don't bind together)

    :param name: the name of a file of the form "xxxx_yyyy[.csv]"
    :return:
    """
    return not (is_positive(name))


def show_progress(iterable):
    """
    Custom progress bar
    :param iterable: the iterable to wrap
    :return:
    """
    return progressbar.progressbar(iterable, widgets=widgets_progressbar, redirect_stdout=True)


def extract_id(file_name):
    """
    Return the id of the file, ie:
        for "xxxx_pro_cg.pdb", "xxxx" is returned

    :param file_name: the name of the file
    :return ID of the file as a string
    """
    return file_name.split("_")[0]


def get_current_timestamp():
    """
    Return the current timestamp of the current time as a string.
    :return:
    """
    utc_dt = datetime.datetime.now(datetime.timezone.utc)
    local_time_dt = utc_dt.astimezone()
    return f"{local_time_dt}".replace(" ", "_")


class LogEpochBatchCallback(keras.callbacks.LambdaCallback):
    """
    Callback to log batch and epoch metrics.

    LambdaCallback are basically wrappers of function that get called after each batch or epoch.

    From Keras documentation : https://keras.io/callbacks/
        > "on_epoch_end: logs include acc and loss, and optionally include val_loss (if validation is enabled in fit)
        > and val_acc (if validation and accuracy monitoring are enabled).
        > on_batch_begin: logs include size, the number of samples in the current batch.
        > on_batch_end: logs include loss, and optionally acc (if accuracy monitoring is enabled)."
    """

    def _on_batch_begin(self, batch, logs=None):
        self.logger.debug(f"batch {batch} ; size {logs['size']}")

    def _on_batch_end(self, batch, logs=None):
        self.logger.debug("loss {:10.4f}".format(logs["loss"]))

    def _on_epoch_begin(self, epoch, logs=None):
        self.logger.debug(f"starting epoch {epoch}")

    def _on_epoch_end(self, epoch, logs=None):
        self.logger.debug(f"ending epoch {epoch}" + " ; accuracy  {:.2%} ; loss {:10.4f}".format(logs["acc"],
                                                                                                 logs["loss"]))

    def __init__(self, logger):
        self.logger = logger
        super().__init__(on_epoch_begin=self._on_epoch_begin,
                         on_epoch_end=self._on_epoch_end,
                         on_batch_begin=self._on_batch_begin,
                         on_batch_end=self._on_batch_end)


def load_nparray(file_name: str):
    """
    Loads an numpy ndarray stored in given file
    :param file_name: the file to use
    :return:
    """

    example = np.loadtxt(file_name, dtype=FLOAT_TYPE, comments=COMMENT_DELIMITER)
    # If it's a vector (i.e if there is just one atom),
    # we reshape it into a (1,n nb_features) array
    if len(example.shape) == 1:
        example = example.reshape(1, -1)

    return example