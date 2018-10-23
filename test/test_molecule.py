import unittest
import numpy as np
import warnings
warnings.simplefilter("ignore")

from code.extraction_data import build_molecule_features


class TestMolecule(unittest.TestCase):
    """
    Testing the construction of molecules independently on one character at the time.

    """

    def setUp(self):
        self.x_list = [1, 2, 3]
        self.y_list = [1, 5, 4]
        self.z_list = [-1, 0, 2]
        self.atom_types = ['C', 'N', 'O']
        self.is_hydrophobic_m = [1, 0, 0]
        self.is_polar_m = [0, 1, 1]
        self.is_protein = True
        self.is_protein_m = [1, 1, 1]
        self.is_ligand_m = [0, 0, 0]

    def test_molecule_from_protein(self):

        is_protein = True
        is_protein_m = [1, 1, 1]
        is_ligand_m = [0, 0, 0]

        extracted_molecule = build_molecule_features(self.x_list, self.y_list, self.z_list, self.atom_types, is_protein)
        molecule_from_protein = np.array([self.x_list, self.y_list, self.z_list,
                                          self.is_hydrophobic_m, self.is_polar_m, is_protein_m, is_ligand_m]).T

        np.testing.assert_array_equal(extracted_molecule, molecule_from_protein)

    def test_molecule_from_ligand(self):

        is_protein = False
        is_protein_m = [0, 0, 0]
        is_ligand_m = [1, 1, 1]

        extracted_molecule = build_molecule_features(self.x_list, self.y_list, self.z_list, self.atom_types, is_protein)
        molecule_from_ligand = np.array([self.x_list, self.y_list, self.z_list,
                                         self.is_hydrophobic_m, self.is_polar_m, is_protein_m, is_ligand_m]).T

        np.testing.assert_array_equal(extracted_molecule, molecule_from_ligand)

    def test_hydrophobic_atoms_only(self):

        atom_types = ['C','C','C']
        is_hydrophobic_m = [1, 1, 1]
        is_polar_m = [0, 0, 0]

        extracted_molecule = build_molecule_features(self.x_list, self.y_list, self.z_list, atom_types, self.is_protein)
        molecule_hydro = np.array([self.x_list, self.y_list, self.z_list,
                                   is_hydrophobic_m, is_polar_m, self.is_protein_m, self.is_ligand_m]).T

        np.testing.assert_array_equal(extracted_molecule, molecule_hydro)

    def test_polar_atoms_only(self):

        atom_types = ['B','N','S']
        is_hydrophobic_m = [0, 0, 0]
        is_polar_m = [1, 1, 1]

        extracted_molecule = build_molecule_features(self.x_list, self.y_list, self.z_list, atom_types, self.is_protein)
        molecule_polar = np.array([self.x_list, self.y_list, self.z_list,
                                   is_hydrophobic_m, is_polar_m, self.is_protein_m, self.is_ligand_m]).T

        np.testing.assert_array_equal(extracted_molecule, molecule_polar)


if __name__ == '__main__':
    unittest.main()
