import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# from src.Experiment import Experiment #.src? 
# from src.SimplifiedThreePL import SimplifiedThreePL

from Experiment import Experiment #.src? 
from SimplifiedThreePL import SimplifiedThreePL

class TestSimplifiedThreePL(unittest.TestCase):

    def create_experiment(self, trials):
        """Creates an Experiment instance and assigns trials dynamically."""
        experiment = Experiment()
        experiment.trials = trials
        return experiment

    def test_constructor_accepts_valid_experiment(self):
        """Test that the constructor correctly accepts a valid Experiment object."""
        experiment = self.create_experiment([])
        obj = SimplifiedThreePL(experiment)
        self.assertIs(obj.experiment, experiment)

    def test_constructor_with_invalid_type_raises_type_error(self):
        """Test that the constructor raises TypeError for non-Experiment objects."""
        invalid_inputs = [None, 42, "invalid", [], {}]

        for invalid_input in invalid_inputs:
            with self.subTest(invalid_input=invalid_input):
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
            ([], 0, 0, 0, 0),
            ([{"correct": True, "condition": "A"}], 1, 1, 0, 1),
            ([{"correct": False, "condition": "A"}], 1, 0, 1, 1),
            ([{"correct": True, "condition": "A"}, {"correct": False, "condition": "B"}], 2, 1, 1, 2),
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

    def test_predict_correct_probabilities(self):
        """Test that predict() returns correct probabilities for given parameters."""
        trials = [
            {"correct": True, "condition": "A"},
            {"correct": False, "condition": "A"},
            {"correct": True, "condition": "B"},
        ]
        experiment = self.create_experiment(trials)
        model = SimplifiedThreePL(experiment)

        parameters = {
            "A": {"a": 1.2, "b": 0.5, "c": 0.2},
            "B": {"a": 0.8, "b": -0.2, "c": 0.1},
        }

        expected_output = {}
        for condition, param in parameters.items():
            a, b, c = param["a"], param["b"], param["c"]
            x = 0  
            prob = c + (1 - c) * (1 / (1 + np.exp(-a * (x - b))))
            expected_output[condition] = round(prob, 4)

        predictions = model.predict(parameters)
        for condition, expected_prob in expected_output.items():
            with self.subTest(condition=condition):
                self.assertAlmostEqual(predictions[condition], expected_prob, places=4)

    def test_predict_handles_empty_experiment(self):
        """Test that predict() returns an empty dictionary when no trials exist."""
        experiment = self.create_experiment([])
        model = SimplifiedThreePL(experiment)
        parameters = {"A": {"a": 1.0, "b": 0.0, "c": 0.2}}
        self.assertEqual(model.predict(parameters), {})
    
    def test_negative_log_likelihood(self):
        """Test that negative_log_likelihood() correctly computes the log likelihood."""
        trials = [
            {"correct": True, "condition": "A"},
            {"correct": False, "condition": "A"},
            {"correct": True, "condition": "B"},
        ]
        experiment = self.create_experiment(trials)
        model = SimplifiedThreePL(experiment)

        parameters = {
            "A": {"a": 1.2, "b": 0.5, "c": 0.2},
            "B": {"a": 0.8, "b": -0.2, "c": 0.1},
        }

        # Compute expected negative log-likelihood manually
        log_likelihood = 0
        for trial in trials:
            condition = trial["condition"]
            correct = trial["correct"]

            # Get predicted probability using the actual function
            prob = model.predict(parameters).get(condition, 0.5)  # Default to 0.5 if missing

            # Compute log likelihood
            if correct:
                log_likelihood += np.log(prob)
            else:
                log_likelihood += np.log(1 - prob)

        expected_nll = -log_likelihood  # Negative log-likelihood

        # Compare model output to expected output
        self.assertAlmostEqual(model.negative_log_likelihood(parameters), expected_nll, places=4)



# Run tests if executed directly
if __name__ == "__main__":
    unittest.main()


