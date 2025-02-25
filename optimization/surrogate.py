import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler

class SurrogateModel:
    def __init__(self):
        # Normalize 
        self.scaler_X = StandardScaler()  
        self.scaler_Y = StandardScaler() 
        self.kernel = Matern(nu=2.5)  # Smooth kernel
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-5, normalize_y=True)

        # Storage for past observations
        self.X_train = None
        self.Y_train = None

    def fit(self, X, Y):
        """ Train the surrogate model with new observations. """
        X_scaled = self.scaler_X.fit_transform(X)
        Y_scaled = self.scaler_Y.fit_transform(Y)

        self.gp.fit(X_scaled, Y_scaled)
        self.X_train, self.Y_train = X_scaled, Y_scaled

    def predict(self, X):
        """ Predicts performance metrics and uncertainty for new hyperparameters. """
        X_scaled = self.scaler_X.transform(X)
        mean, std = self.gp.predict(X_scaled, return_std=True)
        return self.scaler_Y.inverse_transform(mean), std

    def acquisition_function(self, X, strategy="EI"):
        """
        Expected Improvement (EI) or Upper Confidence Bound (UCB) acquisition function
        to guide selection of next hyperparameter candidates.
        """
        mean, std = self.predict(X)

        if strategy == "EI":
            # Expected Improvement
            best_so_far = np.max(self.Y_train[:, 0])  # Max PSNR
            Z = (mean[:, 0] - best_so_far) / (std[:, 0] + 1e-9)
            ei = (mean[:, 0] - best_so_far) * norm.cdf(Z) + std[:, 0] * norm.pdf(Z)
            return ei
        elif strategy == "UCB":
            # Upper Confidence Bound
            beta = 1.96  # Exploration-exploitation trade-off factor
            return mean[:, 0] + beta * std[:, 0]
        else:
            raise ValueError("Unknown acquisition strategy. Choose 'EI' or 'UCB'.")

    def suggest_next(self, candidates):
        """ Selects the next hyperparameter candidate using the acquisition function. """
        acquisition_values = self.acquisition_function(candidates)
        return candidates[np.argmax(acquisition_values)]  # Pick the best candidate
