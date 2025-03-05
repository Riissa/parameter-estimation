#This code was assisted with the help of ChatGPT
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

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
        #changed parameters from A,B to 2,1?
        parameters = {
            "A": {"a": 1.2, "b": 0.5, "c": 0.2},
            "B": {"a": 0.8, "b": -0.2, "c": 0.1},
        }
        ##Modify test later so that condition is a number not a letter?
        # for condition in [2, 1, 0, -1, -2]:  #  Use correct keys
            #self.assertAlmostEqual(predictions[condition], expected_prob, places=4)

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

    def test_negative_log_likelihood_computation(self):
        """Test that negative_log_likelihood correctly computes expected values."""
        
        # Create synthetic experiment data
        trials = [
            {"correct": True, "condition": "A"},
            {"correct": False, "condition": "A"},
            {"correct": True, "condition": "B"},
            {"correct": False, "condition": "B"},
            {"correct": True, "condition": "A"},
        ]
        
        experiment = self.create_experiment(trials)
        model = SimplifiedThreePL(experiment)

        parameters = {
            "A": {"a": 1.2, "b": 0.5, "q": 0.2},  # Using q instead of c
            "B": {"a": 0.8, "b": -0.2, "q": -1.0},  # Logit-transformed c
        }

        # Compute negative log-likelihood
        computed_nll = model.negative_log_likelihood(parameters)

        # Ensure the computed NLL is a valid float
        self.assertIsInstance(computed_nll, float, "NLL is not a float!")
        self.assertGreaterEqual(computed_nll, 0, "NLL should be non-negative")

    def test_negative_log_likelihood_improves_after_fitting(self):
        """Test that negative log-likelihood improves after fitting."""
        
        # Create a set of trials
        trials = [
            {"correct": True, "condition": "A"},
            {"correct": False, "condition": "A"},
            {"correct": True, "condition": "B"},
            {"correct": False, "condition": "B"},
            {"correct": True, "condition": "A"},
        ]
        
        experiment = self.create_experiment(trials)
        model = SimplifiedThreePL(experiment)

        # Initial parameter guess (random or unoptimized)
        initial_parameters = {
            "A": {"a": 1.0, "b": 0.0, "q": 0.0},
            "B": {"a": 0.8, "b": -0.2, "q": 0.0},
        }

        # Compute initial negative log-likelihood
        initial_nll = model.negative_log_likelihood(initial_parameters)
        self.assertIsNotNone(initial_nll, "Initial NLL is None!")

        # Fit the model (this should optimize the parameters)
        model.fit()  # Assuming fit() updates internal parameters

        
        # Compute the new NLL after fitting
        fitted_nll = model.negative_log_likelihood(optimized_parameters)
        self.assertIsNotNone(fitted_nll, "Fitted NLL is None!")

        # Print debugging info
        print(f"Initial NLL: {initial_nll}, Fitted NLL: {fitted_nll}")

        # Check that NLL has improved (decreased)
        self.assertLess(fitted_nll, initial_nll, "NLL did not improve after fitting")
#--------------
    def test_model_stability_and_prediction_accuracy(self):
        """Integration Test Code"""

        # Step 1: Create dataset with five conditions and 100 trials per condition
        accuracy_rates = {2: 0.55, 1: 0.60, 0: 0.75, -1: 0.90, -2: 0.95}
        trials = []

        for condition, accuracy in accuracy_rates.items():
            correct_trials = int(accuracy * 100)  # How many should be correct
            incorrect_trials = 100 - correct_trials  # Remaining should be incorrect

            # Create 100 trials per condition
            trials.extend([{"condition": condition, "correct": True}] * correct_trials)
            trials.extend([{"condition": condition, "correct": False}] * incorrect_trials)

        # Step 2: Create the Experiment instance
        experiment = Experiment()
        experiment.trials = trials

        # Step 3: Fit the model multiple times to check stability
        model = SimplifiedThreePL(experiment)
        first_fit = model.fit()
        second_fit = model.fit()

        # Extract parameters from first and second fit
        first_q, first_c, first_a, first_probabilities = first_fit
        second_q, second_c, second_a, second_probabilities = second_fit

        # Step 4: Verify parameter stability (small deviation allowed)
        self.assertAlmostEqual(first_q, second_q, places=2, msg="q parameter is not stable across fits")
        self.assertAlmostEqual(first_c, second_c, places=2, msg="c parameter is not stable across fits")
        self.assertAlmostEqual(first_a, second_a, places=2, msg="a parameter is not stable across fits")

        # Step 5: Verify that predictions match expected accuracy rates
        for condition, expected_accuracy in accuracy_rates.items():
            predicted_accuracy = first_probabilities[condition]
            self.assertAlmostEqual(
                predicted_accuracy, expected_accuracy, places=2,
                msg=f"Predicted probability {predicted_accuracy} does not match observed accuracy {expected_accuracy} for condition {condition}"
            )

        print(f"Stable parameters: q={first_q}, c={first_c}, a={first_a}")
        print(f"Predictions per condition: {first_probabilities}")





# Run tests if executed directly
if __name__ == "__main__":
    unittest.main()


