import paramiko
import os
import getpass
import pickle
import matplotlib.pyplot as plt

from settings import RESULTS_FOLDER


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


def plot_scores(file=''):
    """
    Plot the F1 score using a serialized history
    :param file:
    :return:
    """
    with open(file, 'rb') as f:
        data = pickle.load(f)

    epoches = list(range(1, len(data['loss']+1)))

    plt.figure()
    plt.plot(epoches, data['f1'], c='black', label='training')
    plt.plot(epoches, data['val_f1'], c='blue', label='evaluation')
    plt.title('F1 scores vs Epoches')
    plt.xlabel('Epoches')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    history_files = download_file()
    for index, file in enumerate(history_files):
        print(f'#{index} :- {file}')
    print('Choose which history you want to plot: ')
    choice = input('Choice: ')
    plot_scores(history_files[int(choice)])
