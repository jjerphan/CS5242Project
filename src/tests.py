import unittest
from extraction_data import extract_molecule
import numpy as np

class MyTests(unittest.TestCase):
    def test_extraction_molecule(self):
        x_list = [1, 2, 3]
        y_list = [1, 5, 4]
        z_list = [-1, 0, 2]
        atom_type = ['C', 'N', 'O']
        is_protein = [True, True, True]

        atom_type_m = [1, 0, 0]
        is_protein_m = [1, 1, 1]

        molecule = extract_molecule(x_list, y_list, z_list, atom_type, is_protein)

        molecule_t = np.array([x_list, y_list, z_list, atom_type_m, is_protein_m])

        assertEqual(molecule, molecule_t)


if __name__ == '__main__':
    unittest.main()