import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))
from src.Experiment import Experiment
from src.SimplifiedThreePL import SimplifiedThreePL

class TestSimplifiedThreePL(unittest.TestCase):

    def create_experiment(self, trials):
        """Helper function to create an Experiment instance with custom trial data."""
        return Experiment(trials)

    def test_constructor_accepts_valid_experiment(self):
        """Test that the constructor correctly accepts a valid Experiment object."""
        experiment = self.create_experiment([])
        obj = SimplifiedThreePL(experiment)
        self.assertIs(obj.experiment, experiment)

    def test_constructor_with_invalid_type_raises_type_error(self):
        """Test that the constructor raises TypeError for non-Experiment objects."""
        for invalid_input in [None, 42, "invalid", [], {}]:
            with self.assertRaises(TypeError):
                SimplifiedThreePL(invalid_input)

    def test_summary_returns_correct_keys(self):
        """Test that summary() returns a dictionary with expected keys."""
        experiment = self.create_experiment([])
        model = SimplifiedThreePL(experiment)
        summary = model.summary()
        expected_keys = {"n_total", "n_correct", "n_incorrect", "n_conditions"}
        self.assertEqual(set(summary.keys()), expected_keys)

    def test_summary_computes_correct_values(self):
        """Test that summary() correctly calculates trial statistics dynamically."""
        test_cases = [
            # Format: (trial data, expected n_total, n_correct, n_incorrect, n_conditions)
            ([], 0, 0, 0, 0),  # No trials
            ([{"correct": True, "condition": "A"}], 1, 1, 0, 1),  # One correct trial
            ([{"correct": False, "condition": "A"}], 1, 0, 1, 1),  # One incorrect trial
            ([
                {"correct": True, "condition": "A"},
                {"correct": False, "condition": "A"},
                {"correct": True, "condition": "B"},
                {"correct": False, "condition": "B"},
                {"correct": True, "condition": "C"},
            ], 5, 3, 2, 3),  # Multiple conditions and correct/incorrect trials
            ([
                {"correct": True, "condition": "X"},
                {"correct": True, "condition": "X"},
                {"correct": False, "condition": "Y"},
                {"correct": False, "condition": "Z"},
                {"correct": True, "condition": "Z"},
                {"correct": False, "condition": "Z"},
            ], 6, 3, 3, 3),  # Another test with multiple conditions
        ]

        for trials, expected_n_total, expected_n_correct, expected_n_incorrect, expected_n_conditions in test_cases:
            with self.subTest(trials=trials):
                experiment = self.create_experiment(trials)
                model = SimplifiedThreePL(experiment)
                summary = model.summary()
                
                self.assertEqual(summary["n_total"], expected_n_total)
                self.assertEqual(summary["n_correct"], expected_n_correct)
                self.assertEqual(summary["n_incorrect"], expected_n_incorrect)
                self.assertEqual(summary["n_conditions"], expected_n_conditions)

# Run tests if executed directly
# if __name__ == "__main__":
#     unittest.main()

