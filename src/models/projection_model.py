import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class ProjectionModel:
    def __init__(self, data):
        """
        Initialize the model with the provided data.
        Args:
            data (numpy.ndarray): Preprocessed fingerprint data.
        """
        self.data = data
        self.model = RandomForestRegressor()

    def train_model(self):
        """
        Train the model using the provided data.
        """
        if not isinstance(self.data, np.ndarray):
            raise ValueError("Data must be a numpy.ndarray.")
        
        if len(self.data.shape) == 3:
            # Flatten the 3D array into a 2D array
            num_samples, height, width = self.data.shape
            X = self.data.reshape(num_samples, height * width)
        else:
            X = self.data[:, :-1]  # All columns except the last
        
        # Generate dummy labels for training (replace with actual labels if available)
        y = np.random.rand(X.shape[0]) * 100  # Simulate labels as random values between 0 and 100
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        print("Model training complete.")

    def predict_age(self, data, age):
        """
        Predict fingerprint changes for a given age.
        Args:
            data (numpy.ndarray): Input data for prediction.
            age (int): Target age for prediction.
        Returns:
            numpy.ndarray: Predicted fingerprint changes.
        """
        if self.model is None:
            raise Exception("Model has not been trained yet. Call train_model() first.")
        
        if len(data.shape) == 3:
            # Flatten the 3D array into a 2D array
            num_samples, height, width = data.shape
            data = data.reshape(num_samples, height * width)
        
        # Simulate predictions by adding random noise based on the age
        predictions = self.model.predict(data)
        noise = np.random.normal(loc=age, scale=5, size=predictions.shape)
        return predictions + noise