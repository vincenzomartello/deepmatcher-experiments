import sys
sys.path.append('..')
import unittest
from utils.triangles_method import getMixedTriangles
import pandas as pd

class TestStringMethods(unittest.TestCase):
    
    def setUp(self):
        lsource = pd.read_csv('test-datasets/Amazon-Google/tableA.csv',dtype=str).fillna("")
        rsource = pd.read_csv('test-datasets/Amazon-Google/tableB.csv',dtype=str).fillna("")
        self.sources = [lsource,rsource]

    def test_triangles_discovery(self):
        zero_triangles = pd.DataFrame(data = {"id":["1#2","10#20"],"label":[1,0]})
        two_triangles = pd.DataFrame(data = {"id":["1#2","3#2","1#7"],"label":[1,0,0]})
        self.assertEqual(len(getMixedTriangles(zero_triangles,self.sources)),0,"there should be zero triangles in this data")
        self.assertEqual(len(getMixedTriangles(two_triangles,self.sources)),2,"there should be two triangles in this data")
        three_triangles = pd.DataFrame(data = {"id":["23#47","23#40","23#50","100#47"],"label":[1,0,0,0]})
        self.assertEqual(len(getMixedTriangles(three_triangles,self.sources)),3,"there should be three triangles in this data")

if __name__ == '__main__':
    unittest.main()