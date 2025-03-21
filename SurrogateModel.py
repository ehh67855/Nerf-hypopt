from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
import numpy as np
from run_nerf import evaluate_hyperparameters

'''
You’re modeling three different objectives: psnr, time, and memory. These metrics:

Have different units and scales (e.g., dB, seconds, megabytes).
May have different relationships to the input parameters.
May be correlated, but not necessarily in a way a single-output GP can capture.
A standard GaussianProcessRegressor from sklearn models only one scalar output at a time, which means:

You can’t directly use a single GaussianProcessRegressor to predict multiple targets (multi-output) unless you heavily customize it.

'''

class SurrogateModel:
    def __init__(self):
        # Initialize a separate Gaussian Process Regressor (GPR) for each objective:
        # PSNR (image quality), time (e.g., training or inference duration), and memory (e.g., peak RAM usage)

        self.gp_psnr = GaussianProcessRegressor(
            kernel=ConstantKernel() * RBF(),          # Use a product of ConstantKernel and RBF (Radial Basis Function) kernel
            n_restarts_optimizer=10,                  # Try optimizing kernel hyperparameters 10 times with different starting points
            random_state=42                           # Set random seed for reproducibility
        )

        self.gp_time = GaussianProcessRegressor(
            kernel=ConstantKernel() * RBF(),          # Same kernel setup for time objective
            n_restarts_optimizer=10,                  # Helps avoid poor local minima during training
            random_state=42
        )

        self.gp_memory = GaussianProcessRegressor(
            kernel=ConstantKernel() * RBF(),          # Same kernel setup for memory objective
            n_restarts_optimizer=10,
            random_state=42
        )

        # StandardScaler will be used to normalize input features (X)
        # Normalization improves numerical stability and GP performance
        self.scaler_X = StandardScaler()

        # Dictionary of StandardScalers, one for each target metric (output variable)
        # These will normalize the output values (PSNR, time, memory) before training the GPs
        self.scaler_y = {
            'psnr': StandardScaler(),
            'time': StandardScaler(),
            'memory': StandardScaler()
        }

        # Placeholder for scaled input training data
        self.X_train = None

        # Dictionary to store scaled output training values for each objective
        self.y_train = {
            'psnr': [],    # Will hold normalized PSNR target values
            'time': [],    # Will hold normalized time target values
            'memory': []   # Will hold normalized memory target values
        }

    def fit(self, X, y_dict):
        """Fit the surrogate models to the training data"""
        # Scale the input features using standard scaling (mean 0, std 1)
        # This helps Gaussian Processes perform better and avoid numerical issues
        self.X_train = self.scaler_X.fit_transform(X)

        # Loop over each target metric: 'psnr', 'time', 'memory'
        for metric, values in y_dict.items():
            # Reshape the values into a column vector and scale them
            # This normalization ensures all targets are on the same scale
            scaled_y = self.scaler_y[metric].fit_transform(np.array(values).reshape(-1, 1))
            
            # Store the flattened scaled values into the y_train dictionary
            self.y_train[metric] = scaled_y.ravel()

        # Fit the Gaussian Process models to the scaled training data
        # Each GP learns to approximate the function for its corresponding metric
        self.gp_psnr.fit(self.X_train, self.y_train['psnr'])
        self.gp_time.fit(self.X_train, self.y_train['time'])
        self.gp_memory.fit(self.X_train, self.y_train['memory'])


    def predict(self, X):
        """Predict psnr, time, and memory using the trained surrogate models"""

        # Scale the input data using the same scaler fitted during training
        X_scaled = self.scaler_X.transform(X)

        # Predict the scaled outputs using each Gaussian Process model
        psnr_pred = self.gp_psnr.predict(X_scaled)
        time_pred = self.gp_time.predict(X_scaled)
        memory_pred = self.gp_memory.predict(X_scaled)

        # Inverse transform the predictions to return them to the original scale
        # This gives human-readable predictions instead of normalized values
        psnr_pred = self.scaler_y['psnr'].inverse_transform(psnr_pred.reshape(-1, 1)).ravel()
        time_pred = self.scaler_y['time'].inverse_transform(time_pred.reshape(-1, 1)).ravel()
        memory_pred = self.scaler_y['memory'].inverse_transform(memory_pred.reshape(-1, 1)).ravel()

        # Return the final predictions for each metric
        return psnr_pred, time_pred, memory_pred
    

if __name__ == "__main__":
    # Define test hyperparameters
    test_hyperparams = {
        'learning_rate': 5e-4,
        'num_hidden_layers': 8,
        'neurons_per_layer': 256,
        'batch_size': 1024,
        'learning_decay_steps': 500
    }

    # Get evaluation results
    psnr, time, mem, _ = evaluate_hyperparameters(test_hyperparams)
    print(f"PSNR: {psnr}, Time: {time}, Memory: {mem}")

    # Create surrogate model
    surrogate = SurrogateModel()

    # Convert dictionary to 2D numpy array for fitting
    X = np.array([[
        test_hyperparams['learning_rate'],
        test_hyperparams['num_hidden_layers'],
        test_hyperparams['neurons_per_layer'],
        test_hyperparams['batch_size'],
        test_hyperparams['learning_decay_steps']
    ]])

    # Fit the model with the reshaped data
    surrogate.fit(X, {
        'psnr': [psnr],
        'time': [time],
        'memory': [mem]
    })

    # Test prediction
    pred_psnr, pred_time, pred_mem = surrogate.predict(X)
    print("\nPredictions for the same hyperparameters:")
    print(f"Predicted PSNR: {pred_psnr[0]:.3f} (Actual: {psnr:.3f})")
    print(f"Predicted Time: {pred_time[0]:.3f} (Actual: {time:.3f})")
    print(f"Predicted Memory: {pred_mem[0]:.3f} (Actual: {mem:.3f})")