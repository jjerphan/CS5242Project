import unittest
import warnings
warnings.simplefilter("ignore")

from src.pipeline_fixtures import is_positive, is_negative, extract_id


class TestFixtures(unittest.TestCase):
    """
    Testing various small fixtures for the pipeline

    """

    def setUp(self):
        self.examples_file_name = ["0001_0001.csv", "0001_0002.csv", "0000_0000.csv",
                                   "1841_1245.csv", "0001_0000.csv", "1314_1314.csv"]

    def test_is_positive(self):
        positivity = [True, False, True, False, False, True]
        self.assertTrue(positivity, list(map(is_positive, self.examples_file_name)))

    def test_is_negative(self):
        negativity = [False, True, False, True, True, False]
        self.assertTrue(negativity, list(map(is_negative, self.examples_file_name)))

    def test_extract_id(self):
        to_extract = ["0000_pro_cg.pdb,7_pro_cg.pdb,17482891_pro_cg.pdb",
                      "1373_lig_cg.pdb,1114_lig_cg.pdb,0_lig_cg.pdb"]
        ids = ["0000", "7", "17482891", "1373", "1114", "0"]
        self.assertTrue(ids, list(map(extract_id, to_extract)))
