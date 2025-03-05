import numpy as np
from Experiment import Experiment  # Ensure correct import

class SimplifiedThreePL:
    def __init__(self, experiment):
        """Initialize with an Experiment instance."""
        if not isinstance(experiment, Experiment):
            raise TypeError("Expected an Experiment instance")
        self.experiment = experiment

    def summary(self):
        """Returns a dictionary summarizing the experiment's trial data."""
        if not hasattr(self.experiment, 'trials') or self.experiment.trials is None:
            return {
                "n_total": 0, 
                "n_correct": 0, 
                "n_incorrect": 0, 
                "n_conditions": 0
            }

        return {
            "n_total": len(self.experiment.trials),
            "n_correct": sum(1 for trial in self.experiment.trials if trial.get("correct", False)),
            "n_incorrect": sum(1 for trial in self.experiment.trials if not trial.get("correct", False)),
            "n_conditions": len(set(trial["condition"] for trial in self.experiment.trials if "condition" in trial)),
        }

    def predict(self, parameters):
        """Returns the probability of a correct response in each condition, given the parameters."""
        if not hasattr(self.experiment, 'trials') or not self.experiment.trials:
            return {}

        probabilities = {}
        unique_conditions = set(trial["condition"] for trial in self.experiment.trials if "condition" in trial)

        for condition in unique_conditions:
            a = parameters.get(condition, {}).get("a", 1)  # Default a=1
            b = parameters.get(condition, {}).get("b", 0)  # Default b=0
            c = parameters.get(condition, {}).get("c", 0)  # Default c=0
            x = 0  # Assume ability level (theta) = 0

            probability = c + (1 - c) * (1 / (1 + np.exp(-a * (x - b))))
            probabilities[condition] = round(probability, 4)  # Match rounding in test

        return probabilities
    import numpy as np

    def negative_log_likelihood(self, parameters):
        """Computes the negative log-likelihood of the data given the parameters."""
        if not hasattr(self.experiment, 'trials') or not self.experiment.trials:
            return 0  # If there are no trials, return 0 (no likelihood to compute)

        log_likelihood = 0

        for trial in self.experiment.trials:
            condition = trial["condition"]
            correct = trial["correct"]

            # Get predicted probability for this

    



    

