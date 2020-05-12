import unittest
from blocking import get_cosine,findPossibleMatchings
import pandas as pd

class TestStringMethods(unittest.TestCase):
    
    def setUp(self):
        self.text1 = "sony bravia"
        self.text2 = "sony bravia"
        self.text3 = "samsung 4k"
        self.source = pd.read_csv('test-datasets/Amazon-Google/tableA.csv',dtype=str).fillna("")
        self.record = self.source.iloc[0]

    def test_cosine_sim(self):
        self.assertTrue(get_cosine(self.text1,self.text2)>=0.99,"cosine similarity should be 1")
        self.assertTrue(get_cosine(self.text1,self.text3)<=0)
        
    def test_possible_matchings_search(self):
        possibleMatchings = findPossibleMatchings(self.record,self.source,0.1)
        self.assertTrue(len(possibleMatchings)>=0)
        self.assertEqual(len(possibleMatchings.columns),2,"resulting dataframe should have 2 columns")

if __name__ == '__main__':
    unittest.main()