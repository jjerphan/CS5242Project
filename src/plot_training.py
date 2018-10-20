import paramiko
import getpass
import pickle
import matplotlib.pyplot as plt


def download_file():
    hostname = 'nus.nscc.sg'
    username = input("Input your student id: ")
    basedir = '/home/users/nus/' + str(username) + '/'
    localfile = './results/historys/history.pickle'

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=hostname, username=username, password='')

    input, output, error = ssh.exec_command('ls ' + basedir + '*/results/*/history.pickle')

    for line in output:
        print(line)
    #ftp_client = ssh.open_sftp()
    #ftp_client.get(remotefile, localfile)
    #ftp_client.close()

    ssh.close()

def plot_scores(file='./results/history.pickle'):
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
    plot_scores()