import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))
from src.Experiment import Experiment
from src.SimplifiedThreePL import SimplifiedThreePL


class TestSimplifiedThreePL(unittest.TestCase):
    
    def setUp(self):
        """Set up a sample Experiment object for testing."""
        self.experiment = Experiment()
        self.model = SimplifiedThreePL(self.experiment)
    
    def test_constructor_initialization(self):
        """Test that SimplifiedThreePL initializes correctly."""
        model = SimplifiedThreePL(self.experiment)
        
        # Check that the object is an instance of SimplifiedThreePL
        self.assertIsInstance(model, SimplifiedThreePL)

        # Check that the experiment attribute is correctly assigned
        self.assertEqual(model.experiment, self.experiment)

        # Ensure that the model is not fitted initially
        self.assertFalse(model._is_fitted, "Model should not be fitted upon initialization.")

if __name__ == '__main__':
    unittest.main()
