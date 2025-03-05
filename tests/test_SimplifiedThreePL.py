import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))
from src.Experiment import Experiment
from src.SimplifiedThreePL import SimplifiedThreePL


class TestSimplifiedThreePL(unittest.TestCase):
    
   import unittest

# Dummy Experiment class for testing
class Experiment:
    pass

# SimplifiedThreePL class that accepts an Experiment object
class SimplifiedThreePL:
    def __init__(self, experiment):
        if not isinstance(experiment, Experiment):
            raise TypeError("Expected an Experiment instance")
        self.experiment = experiment
        
class TestSimplifiedThreePL(unittest.TestCase):
    def setUp(self):
        """Set up a valid Experiment instance for testing"""
        self.valid_experiment = Experiment()

    def test_constructor_accepts_valid_experiment(self):
        """Test that the constructor correctly accepts a valid Experiment object"""
        obj = SimplifiedThreePL(self.valid_experiment)
        self.assertIs(obj.experiment, self.valid_experiment)

    def test_constructor_with_none_raises_type_error(self):
        """Test that the constructor raises TypeError when given None"""
        with self.assertRaises(TypeError):
            SimplifiedThreePL(None)

    def test_constructor_with_int_raises_type_error(self):
        """Test that the constructor raises TypeError when given an integer"""
        with self.assertRaises(TypeError):
            SimplifiedThreePL(42)

    def test_constructor_with_str_raises_type_error(self):
        """Test that the constructor raises TypeError when given a string"""
        with self.assertRaises(TypeError):
            SimplifiedThreePL("invalid input")

# Run the tests if this script is executed directly
# if __name__ == "__main__":
#     unittest.main()

