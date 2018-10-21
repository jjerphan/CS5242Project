import paramiko
import os
import getpass
import pickle
import matplotlib.pyplot as plt
from settings import results_folder


def download_file():
    hostname = 'nus.nscc.sg'
    username = input("Input your student id: ")
    password = getpass.getpass(prompt='Enter your password: ')
    username = 'e0319586'
    password = 'Haha39462070*'
    basedir = '/home/users/nus/' + str(username) + '/'
    localdir = './results/historys/'
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
    with open(file, 'rb') as f:
        data = pickle.load(f)

    epoches = [i + 1 for i in range(len(data['loss']))]

    plt.figure()
    plt.plot(epoches, data['acc'], c='black', label='training')
    plt.plot(epoches, data['val_acc'], c='blue', label='evaluation')
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
