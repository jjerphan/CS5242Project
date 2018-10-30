Papers
======

Here are different papers to get inspired:

 - [Development and evaluation of a deep learning model for protein–ligand binding affinity prediction](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/bty374/4994792)
    - Interesting papers: gives the intuition as well as the desing of the network
    - Use 3D convolution on 20 Å³ cube with a resolution of 1 Å (yielding 20³ little cubes)
    - 19 features per cube
    - Paphnucy : the implementation made using Tensorflow:
        - trained using the PDBbind database ;
        - implementation available online here 

 - [DLSCORE: A Deep Learning Model for Predicting Protein-Ligand Binding Affinities](https://chemrxiv.org/articles/DLSCORE_A_Deep_Learning_Model_for_Predicting_Protein-Ligand_Binding_Affinities/6159143)
    - Pretty recent paper
    - Apparently have good results
    - Using ensemble methods on N=10 simple neural network (fully-connected layers)
    - Usage of BINANA descriptors to extract information from `.pdb` files:
        - **May make this model impossible to use has we only need to use the x,y,z coordinates and the atom types and molecules types**
        - See the original paper [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3099006/)
        - See an implementation [here](https://github.com/oddt/oddt/blob/master/oddt/scoring/descriptors/binana.py)
    - The implementation is available online on GitHub [here](https://github.com/sirimullalab/)
        - the the back-end of the model is present in the`dlscore` repo
        - the testing procedure is in the `CADD_Workflows` repo

 - [AtomNet: A Deep Convolutional Neural Network for Bioactivity Prediction in Structure-based Drug Discovery](https://arxiv.org/abs/1510.02855)
    - One of the first model CNN model (2015), yield classification problem on 1/0 (bind, don't bind)
    - Data get discretized on the 3D cube as well
 - [Protein-Ligand Scoring with Convolutional Neural Networks](https://arxiv.org/abs/1612.02751)
 - [DeepDTA: Deep Drug-Target Binding Affinity Prediction](https://arxiv.org/abs/1801.10193v2)
    - to explore : might not be related or relevant for what we are doing

