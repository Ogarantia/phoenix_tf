import unittest
import tensorflow as tf
import numpy as np
from modules.upstride_tf   import generic_layers
from tests.test import *

class Test(unittest.TestCase):
    def leader_test(self):
        print("")
        TestGAMultiplication(self)
        
        print("TestQuaternionTF2Upstride")
        TestQuaternionTF2Upstride(self)
        
        print("TestQuaternionMult")
        TestQuaternionMult(self)
        
        print("TestTF")
        TestTF(self)
        
        print("TestType1")
        TestType1(self)
        
        print("TestType2")
        TestType2(self)
        
        print("TestType3")
        TestType3(self)

if __name__ == "__main__":
    unittest.main()
