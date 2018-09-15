# CS5242 Project : Predicting Protein – Ligand Interaction by using Deep Learning Models

## Description of the project

**See [project_description.pdf](./project_description.pdf) for the full description of the project **

### Data Sets given — Notation and formalisation

An example is a pair of a protein and its associated ligand.

For an example numebered `xxxx`, each molecules of the pair is represented in a file:

-  `xxxx_pro_cg_.pdb` for the protein
-  `xxxx_lig_cg_.pdb` for the ligand

There is a **one-to-one association between proteins and ligands** ie: 

​			`xxxx_lig_cg_.pdb` **cannot bind** `xxxx_pro_cg_.pdb`

Each of those files contains information about the atom of the molecules ; for each atom there is **4 features**

- $(x,y,z) \in \mathbb{R}^3$ : its position in 3D
-  $type \in \{hydrophobic, polar \}$

**See comments bellow about "*Data and training*" for an overwiew of data** 

### Goal and evaluation

Find, for a each protein, *the* ligand that binds it.

This is a hard problem, thus the evaluation is eased:

​	For each new protein, find 10 ligens that could fit in the protein.

If one on the match is in the set of the 10 ligens: the answer is considered good  !

The final score is proportional to the total number of match for proteins.

### Some related questions or ideas to experiment

###### Rotations of proteins: 

- Would performing 3D rotations be usefull to have more examples ? 
- It is possible to find a canonical representation for molecules ? I.E for a given protein in a random position, is it possible to find a "natural" position of this same protein easily for every random position ?
  - It should be possible using PCA on the system of protein and ligand, yes!

###### Proteins and their atoms:

- Is there a specific fixed number of atoms for each protein?
  -  No : molecules do have differents number of atoms (see bellow). Generally, proteins have much more atoms than ligands.
- Why considering only 2 types of atoms (*hydrophobic* and *polar*) instead of the 3 first types (*C*, *O* and *N*) ?

###### Data and training

- Where is the data coming from? What are the other features present in the files?
  - It should be coming from the popular [PDBBind](http://www.pdbbind.org.cn/) data base.
- For now, we have $n=3000$ examples (pair of protein and ligens). But if we see the problem as a classification problem from a couple protein/ligen to a prediction fit / not fit, we can construct loads of differents examples. More expecially, we can for each example construct $n-1$ other examples. Those examples won't represent a protein-ligand system, but can be used as example for the "not-fit" class. Hence we could have in total $3000 \times 2999 = 8\ 997\ 000$ examples
- **Bigest problem for now:** the number of atoms is extremely variant for the molecules. See [`info`](./info) .

```
# Number of atoms for proteins and ligands
	pro             lig         
 Min.   :    0   Min.   :  1.000  
 1st Qu.:  546   1st Qu.:  4.000  
 Median :  854   Median :  5.000  
 Mean   : 1343   Mean   :  8.502  
 3rd Qu.: 1577   3rd Qu.:  9.000  
 Max.   :14102   Max.   :300.000 
```

![Density estimation](./info/density.png)

We need to find a way to resolve this problem. There are [several approaches](https://ai.stackexchange.com/questions/2008/how-can-neural-networks-deal-with-varying-input-sizes):

> - Zero padding. 
> - [RNNs](https://en.wikipedia.org/wiki/Recurrent_neural_network)
> - Another possibility is using [recursive NNs](https://en.wikipedia.org/wiki/Recursive_neural_network).



- A (naîve ?) but working solution is to discretize the space containing the protein-ligand system and then represented on each little cube the information of atom. For a given cube, if there is no atom, then, the information is nulled. This approach is used in different models (such as AtomNet and Pafnucy).

  ​

###### Curious observations about the data set   

- Some files do not contain atoms, the molecules are empty, even for some proteins.
  - We should identify wich are those files exactly and then think about how we could handle those missing examples

###### Some others thoughts

- When we are going to test, the approach would be to return the ligands with the highest probability. However, we know that there is an extra constraint on atom, more precisely that there is a one to one correspondance. Hence, we *should or must* take decisions for ligands generally and not per case as we could choose several ligands for several protein with an high confidence. If we are given $n_p$ proteins and $n_l$ ligands to test :
  - The first (naiver) approach would consist to evaluate the $n_l$ ligands and take the 10 best ones.
  - The second approach would consist to evaluate the $n_p\times n_l$ systems and then take, for each ligands that are chosen several times, the associated protein of highest confidence.



## Things to do

- [ ] Do some research on proteins-ligands binding
- [ ] Find appropriate computing ressources
      - [ ] AWS
      - [ ] Google Cloud
      - [ ] NUS Clusters
      - [ ] [NSCC](https://help.nscc.sg/)
- [ ] Create final pipeline to test binding (as what's required)
      - [ ] Use better metrics (Confusion Matrix, F1 Score, ROC)


- [ ] Find a way to balance scores
- [ ] Find goods numbers to scores

## Some resources

- [Protein on Wikipedia](https://en.wikipedia.org/wiki/Protein)
- [Protein Structure on Wikipedia](https://en.wikipedia.org/wiki/Protein_structure)
- [Drug Design on Wikipedia](https://en.wikipedia.org/wiki/Drug_design)
- [Format of `pdb` files](ftp://ftp.wwpdb.org/pub/pdb/doc/format_descriptions/Format_v33_A4.pdf)
- [Download PyMol](https://pymol.org/2/#download)
- [Doc of PyMol](http://pymol.sourceforge.net/newman/userman.pdf)
- [Keras Documentation - Sequence Model](https://keras.io/getting-started/sequential-model-guide/)
- [PDBBind](http://www.pdbbind.org.cn/)