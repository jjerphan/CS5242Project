# NSCC Cluster 

### *Enroll into NSCC cluster*

##### Step 1: Obtain an NSCC Account 

- Go to https://user.nscc.sg/saml

- Login:
    - NUS
    - Enter username/password
    - Yes, continue
    - "Your NSCC account is e0319586é
    - Click "Reset SSH Key"
    - Copy the public key
    - `# touch ~/.ssh/id_rsa`
    - `# chmod 600 ~/.ssh/id_rsa`
    - And paste the key into `~/.ssh/id_rsa`
    - Click "Set/Reset Password"
    - Enter password
    - Done


##### Step 2: Access ASPIRE 1 login nodes

- From NUS 

  - ssh -i ~/.ssh/id_rsa exxxxxxx@nus.nscc.sg 

  - Alternatively create SSH config 
    ```
    Host NSCC nus.nscc.sg
    ​    HostName nus.nscc.sg
    ​    IdentityFile ~/.ssh/id_rsa
    ​    User exxxxxxxx
    ```
  - Then ssh to nscc04-ib0

    `ssh nscc04-ib0`

- From outside using VPN

  - Install VPN client for Linux

    sudo apt-get install network-manager-openvpn-gnome

  - Site: https://help.nscc.sg/vpnmac/



##### Step 3: Getting the environment ready

* Download Anacoda3 
    ```bash
    scp Anaconda3-5.2.0-Linux-x86_64.sh NSCC:~/
    scp Anaconda3-5.2.0-Linux-x86_64.sh nscc04-ib0:~/
    chmod +x Anaconda3-5.2.0-Linux-x86_64.sh
    bash Anaconda3-5.2.0-Linux-x86_64.sh
    ```


### *Submitting jobs*

PBS Pro - schedule jobs on the cluster

Must submit jobs to one of the external queues. 

| Queue Name | Resources Available                                          | Remarks                                                      |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| normal     | 12 Compute nodes 24 cores per server4GB/core memory or 96 GB per server | The queue normal is a routing queue and does not execute any jobs. It only routes the job to the internal queues based on the resource requirement |
| gpu        | 128 GPU nodes24 Cores per server4GB/core memory or 96 GB memory per server | This queue is a routing queue and does not execute any jobs. This queue routes the jobs to the internal queues based on resource requirement |
| largemem   | 9 Compute nodes24/48 cores per server1TB/4TB/6TB memory configurations | This is an execution queue which can take large memory jobs which requires more than 96GB of memory per server |
| production | 480 compute cores reserved for GIS                           | Only selected GIS users can submit jobs to this queue        |
| provision  | Special Administrative queue                                 | No access to users                                           |
| iworkq     | Special queue for visualization jobs                         | Jobs submitted from Display manager to be dispatched to this queue |
| imeq       | Special queue for ime jobs                                   | This queue is under testing                                  |

PBS quick reference sheet: https://help.nscc.sg/wp-content/uploads/2016/08/PBS_Professional_Quick_Reference.pdf

PBS Pro quick start guide: https://help.nscc.sg/pbspro-quickstartguide/

##### Example from quick start guide:

- Submit using pipe 

```bash
echo sleep 10 | qsub -q normal -l select=1:ncpus=1,walltime=00:13:00 
\#\# 1 chunk with 1 cpus, total 13 minutes of wall time 
```


- Submit using job submission script

```bash
$ cat submit.pbs
#! /bin/bash
#PBS -q normal
#PBS -l select=1:ncpus=1:mem=100M		# request for resources
#PBS -l walltime=00:10:00 		# request for resources
#PBS -P Personal 		# the project name
#PBS -N Sleep_Job 		# the name of the job 
#PBS -o ~/outputfiles/Sleep_Job.o		# the output file
#PBS -e ~/errorfiles/Sleep_Job.e 		# the error file 
echo sleep job for 30 seconds
sleep 30 
##

$ qsub submit.pbs
```


- Example submitting CS5242 jobs
```bash
$ export PBS_O_WORKDIR=`pwd`
$ cat training.pbs
#! /bin/bash
#PBS -q gpu
#PBS -j oe
#PBS -l select=1:ngpus=1
#PBS -l walltime=23:00:00 
#PBS -N CS5242_Hello 
cd ${PBS_O_WORKDIR}
source activate CS5242
python -c "import keras; print('if you see no errors, it\'s good to go.')"
```


### *Checking job status*

* See output in <job_name>.oXXXXX

* Run qstat command 
  ```bash
  qstat -x 
  qstat -f \<jobid\>
  ```


### *Submit our project job*

* Create/enter conda environment 

  $ conda create -n CS5242 python=3.6

  $ source activate CS5242

* Install libraries

  $ conda install tensorflow-gpu keras progressbar2 matplotlib tensorflow

* Create submit script
  ```bash
  $  cat training.pbs 
  #! /bin/bash
  #PBS -q gpu 
  #PBS -j oe
  #PBS -l select=1:ngpus=1
  #PBS -l walltime=00:10:00
  #PBS -N CS5242_Training
  cd $(PBS_O_WORKDIR)
  source activate CS5242
  python CNN.py
  ```
  
* Submit the job
  ```bash
  $ export PBS_O_WORKDIR=\`pwd\`
  $ qsub training.pbs
  ```

* Check job execution status
  ```bash
  (CS5242) [exxxxxxx@nscc04 ~]$ qstat -f 7669002.wlm01
  qstat: 7669002.wlm01 Job has finished, use -x or -H to obtain historical job information
  (CS5242) [exxxxxxx@nscc04 ~]$ qstat -x
  Job id            Name             User              Time Use S Queue

  ----------------  ---------------- ----------------  -------- - -----
  7669002.wlm01     CS5242_Training  e0319586          00:00:00 F gpunormal
  ```

* Check queue information 
  ```bash
  $ qstat -x
  ```

* Check job execution output
  ```bash
  $ cat CS5242_Training.o7669473
  ```

