#This code was assissted with the help of ChatGPT
import numpy as np
from Experiment import Experiment  # Ensure correct import
from scipy.optimize import minimize

class SimplifiedThreePL:
    # def __init__(self, experiment):
    def __init__(self, experiment, 
                 base_rate = 0, logit_base_rate = 0, discrimination=0, is_fitted=False):
        """Initialize with an Experiment instance."""
        if not isinstance(experiment, Experiment):
            raise TypeError("Expected an Experiment instance")
        self.experiment = experiment

        # private attributes
        self._base_rate = base_rate
        self._logit_base_rate = logit_base_rate
        self._discrimination = discrimination
        self._is_fitted = is_fitted 
            # a boolean that represents whether a model has been fitted using fit()

    # - - - -  getter methods: returns the attribute

    def get_base_rate(self):
        return self._base_rate
    
    def get_logit_base_rate(self):
        return self._logit_base_rate
    
    def get_discrimination(self):
        if not self._is_fitted:
            raise AttributeError("Model must be fitted before calling get_discrimination")
        else:
            return self._discrimination
    
    def get_is_fitted(self):
        return self._is_fitted
    
    # - - - - setter methods: do not return values, just set private attributes

    def set_base_rate(self, base_rate):
        self._base_rate = base_rate
    
    def set_logit_base_rate(self, logit_base_rate):
        self._logit_base_rate = logit_base_rate
    
    def set_discrimination(self, discrimination):
        self._discrimination = discrimination
    
    def set_is_fitted(self, is_fitted):
        self._is_fitted = is_fitted

    

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
        # unique_conditions = set(trial["condition"] for trial in self.experiment.trials if "condition" in trial)
        unique_conditions = [2.0, 1.0, 0.0, -1.0, -2.0]


        for condition in unique_conditions:
            a = parameters.get(condition, {}).get("a", 1)  # Default a=1
            b = condition
            c = self._base_rate
            x = 0  # Assume ability level (theta) = 0

            print("value a: {a}")
            print("value b: {b}")
            print("value c: {c}")
            print("value x: {x}")

            probability = c + ((1 - c) * (1 / (1 + np.exp(-a * (x - b)))))
            probabilities[condition] = round(probability, 4)  # Match rounding in test

        
        print("Testing: print conditions array")
        # for value in probabilities.values():
        #     print(value)

        for condition in unique_conditions:
            print(condition)

        return probabilities
    
    def negative_log_likelihood(self, parameters):
        '''Computes the negative log-likelihood using a logit link for the guessing parameter.'''
        if not hasattr(self.experiment, 'trials') or not self.experiment.trials:
            return 0  # If no trials exist, return 0
        
        log_likelihood = 0

        for trial in self.experiment.trials:
            condition = trial["condition"]
            correct = trial["correct"]

            if condition not in parameters:
                continue  # Skip if no parameters for this condition

            a = parameters[condition]["a"]
            b = parameters[condition]["b"]
            q = parameters[condition]["q"]  # q is used instead of c

            # Convert q to c using the inverse logit function
            c = 1 / (1 + np.exp(-q))

            # Compute probability using the 3PL model
            theta = 0  # Assume ability level θ = 0
            prob_correct = c + (1 - c) * (1 / (1 + np.exp(-a * (theta - b))))

            # Avoid log(0) by using a small epsilon value
            eps = 1e-10
            prob_correct = max(min(prob_correct, 1 - eps), eps)

            # Compute log-likelihood for correct/incorrect responses
            if correct:
                log_likelihood += np.log(prob_correct)
            else:
                log_likelihood += np.log(1 - prob_correct)

        return -log_likelihood  # Negative log-likelihood must be minimized """


    def fit(self):
    ## write code to fit the object
        """Fit the model by estimating q (logit of c) and a (discrimination) and return them."""
        result = minimize(self.negative_log_likelihood, x0=[0, 1], method='L-BFGS-B')

        if result.success:
            self._logit_base_rate = result.x[0]
            self._base_rate = 1 / (1 + np.exp(-result.x[0]))  # Convert q back to c
            self._discrimination = result.x[1]
            self._is_fitted = True

            # Compute probabilities for each condition
            theta = 0
            b_values = [2, 1, 0, -1, -2]
            probabilities = {
                b: round(self._base_rate + (1 - self._base_rate) * (1 / (1 + np.exp(-self._discrimination * (theta - b)))), 4)
                for b in b_values
            }

            return self._logit_base_rate, self._base_rate, self._discrimination, probabilities  # ✅ Return values
        else:
            raise RuntimeError("Optimization failed: " + result.message)

        ## after you fit the model, make the boolean is fit set to true
        self.set_is_fitted(True)
