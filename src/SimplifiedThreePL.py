from Experiment import Experiment
import numpy as np
#placeholder code to test the tests
class SimplifiedThreePL:
    def __init__(self, experiment):
        if not isinstance(experiment, Experiment):
            raise TypeError("Expected an Experiment instance")
        self.experiment = experiment

    def summary(self):
        """Returns a dictionary summarizing the experiment's trial data."""
        if not hasattr(self.experiment, 'trials'):
            raise AttributeError("Experiment object must have a 'trials' attribute.")

        return {
            "n_total": len(self.experiment.trials),
            "n_correct": sum(1 for trial in self.experiment.trials if trial["correct"]),
            "n_incorrect": sum(1 for trial in self.experiment.trials if not trial["correct"]),
            "n_conditions": len(set(trial["condition"] for trial in self.experiment.trials))
        }
    def predict(self, parameters):
        """Returns the probability of a correct response in each condition, given the parameters."""
        if not hasattr(self.experiment, 'trials') or not self.experiment.trials:
            return {}

        probabilities = {}
        unique_conditions = set(trial["condition"] for trial in self.experiment.trials)

        for condition in unique_conditions:
            a = parameters.get(condition, {}).get("a", 1)  # Default to 1 if missing
            b = parameters.get(condition, {}).get("b", 0)  # Default to 0 if missing
            c = parameters.get(condition, {}).get("c", 0)  # Default to 0 if missing
            x = 0  # Assume ability level (theta) is 0

            probability = c + (1 - c) * (1 / (1 + np.exp(-a * (x - b))))
            probabilities[condition] = round(probability, 4)  # Round for readability

        return probabilities

    

