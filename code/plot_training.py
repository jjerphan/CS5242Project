import paramiko
import os
import getpass
import pickle
import matplotlib.pyplot as plt
import re

from models_inspector import ModelsInspector
from settings import RESULTS_FOLDER, EVALUATION_LOGS_FOLDER


def download_file():
    """
    Download the histories present remotely
    :return:
    """
    hostname = 'nus.nscc.sg'
    username = input("Input your student id: ")
    password = getpass.getpass(prompt='Enter your password: ')
    basedir = '/home/users/nus/' + str(username) + '/'
    localdir = os.path.join(RESULTS_FOLDER, "histories")
    localfiles = []

    if not os.path.exists(localdir):
        os.makedirs(localdir)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=hostname, username=username, password=password)

    In, Out, Err = ssh.exec_command('ls ' + basedir + '*/results/*history.pickle')

    ftp_client = ssh.open_sftp()
    for remotefile in Out:
        remotefile = remotefile.rstrip('\r|\n')
        localfile = os.path.join(localdir, remotefile.split('/')[-1])
        localfiles.append(localfile)
        ftp_client.get(remotefile, localfile)
    ftp_client.close()

    ssh.close()

    return localfiles


def plot_f1_scores(file='', xlim_max=None):
    """
    Plot the F1 score using a serialized history
    :param file:
    :return:
    """
    with open(file, 'rb') as f:
        data = pickle.load(f)

    epoches = list(range(1, len(data['loss']) + 1))

    plt.figure()
    plt.plot(epoches, data['f1'], c='black', label='training')
    plt.plot(epoches, data['val_f1'], c='blue', label='evaluation')
    plt.xlim((0.0, xlim_max))
    plt.ylim((0.0, 1.1))
    plt.title('F1 scores vs Epoches')
    plt.xlabel('Epoches')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid()
    plt.show()


def plot_losses_values(file='', xlim_max=None, ylim_max=None, description=""):
    """
    Plot the F1 score using a serialized history
    :param file:
    :return:
    """
    with open(file, 'rb') as f:
        data = pickle.load(f)

    epoches = list(range(1, len(data['loss']) + 1))

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(epoches, data['loss'], c='black', label='training')
    ax.plot(epoches, data['val_loss'], c='blue', label='evaluation')
    ax.set_xlim((0.0, xlim_max))
    ax.set_ylim((0.0, ylim_max))
    ax.set_title(description)
    ax.set_xlabel('Epoches')
    ax.set_ylabel('Loss')
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(50, 100, 750, 500)
    plt.legend()
    plt.grid()
    plt.show()
    return fig


def plot_remote_results():
    """
    Plot histories of results that are present remotely.

    Download the file locally then ask for the one to plot them.
    """
    history_files = download_file()

    for index, file in enumerate(history_files):
        print(f'#{index} :- {file}')
    print('Choose which history you want to plot: ')
    choice = input('Choice: ')
    plot_f1_scores(history_files[int(choice)])


def plot_local_results():
    """
    Plot histories of results that are present locally.

    Plot all the history one after the other.
    :return:
    """
    models_inspector = ModelsInspector(results_folder=RESULTS_FOLDER, show_without_serialized=True)
    plt.ion()
    for sub_folder, set_parameters, _, history_file_path, _ in models_inspector:
        print(f"Model {sub_folder}")
        for k, v in set_parameters.items():
            print(f" - {k} : {v}")

        optimizer_name = re.sub(r'object at .+>', "", set_parameters['optimizer'].replace('<keras.optimizers.',''))
        description = f"{set_parameters['model']} Repr. {set_parameters['representation']} ; " \
                      f"Epochs {set_parameters['nb_epochs']} ;  " \
                      f"Neg: {set_parameters['nb_neg']} " \
                      f"Weight Pos : {set_parameters['weight_pos_class']} ; " \
                      f"{optimizer_name}"
        fig = plot_losses_values(history_file_path, xlim_max=30, ylim_max=1, description=description)
        plt.pause(1)
        name_file = re.sub(r"\s+", "_", re.sub(r'(\.|;|:)', "", description)).lower()
        fig.savefig(os.path.join(EVALUATION_LOGS_FOLDER, name_file))
        # input("[Enter] to continue")
        plt.close('all')


if __name__ == '__main__':
    plot_local_results()
