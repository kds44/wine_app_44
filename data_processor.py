import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logger = logging.getLogger(__name__)

class WineQualityProcessor:
    """
    A class for processing wine quality data and making predictions.
    
    This class handles:
    - Data loading and preprocessing
    - Feature engineering
    - Model training and evaluation
    - Quality predictions
    - Data visualization preparation
    
    Attributes:
        data (pd.DataFrame): The processed wine quality dataset
        model (RandomForestRegressor): The trained machine learning model
        scaler (StandardScaler): Scaler for normalizing features
        feature_columns (list): List of features used for prediction
    """
    
    def __init__(self):
        """
        Initializes the WineQualityProcessor with empty data and model.
        Sets up the feature columns and scaler for data processing.
        """
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        # All features from the wine quality dataset
        self.feature_columns = [
            'fixed acidity',
            'volatile acidity',
            'citric acid',
            'residual sugar',
            'chlorides',
            'free sulfur dioxide',
            'total sulfur dioxide',
            'pH',
            'sulphates',
            'alcohol'
        ]
        
    def load_data(self):
        """
        Loads the wine quality dataset from a local CSV file.
        
        Returns:
            bool: True if data loading was successful, False otherwise
            
        Note:
            Expects 'winequality-red.csv' to be in the working directory
        """
        try:
            # Load red wine dataset from local CSV file
            self.data = pd.read_csv("winequality-red.csv", sep=";")
            logger.info(f"Dataset loaded successfully: {self.data.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False

    def clean_and_feature_engineer(self):
        """
        Performs data cleaning and feature engineering on the wine dataset.
        
        This function:
        - Removes duplicate entries
        - Handles missing values
        - Removes outliers using IQR method
        
        Returns:
            bool: True if cleaning was successful, False otherwise
            
        Note:
            Uses IQR method for outlier detection on all features
        """
        if self.data is None:
            return False
            
        try:
            # Remove duplicates
            initial_rows = len(self.data)
            self.data = self.data.drop_duplicates()
            logger.info(f"Removed {initial_rows - len(self.data)} duplicate rows")
            
            # Handle missing values (if any)
            self.data = self.data.dropna()
            
            # Remove outliers using IQR method for all features
            for col in self.feature_columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
            
            logger.info(f"Data cleaning completed. Final dataset shape: {self.data.shape}")
            return True
        except Exception as e:
            logger.error(f"Error in data cleaning: {e}")
            return False
    
    def get_descriptive_stats(self):
        """
        Calculates and returns descriptive statistics for the dataset.
        
        Returns:
            dict or None: Dictionary containing:
                - avg_alcohol_by_quality: Average alcohol content by quality rating
                - correlation_matrix: Correlation between features
                - basic_stats: Basic statistical measures
            Returns None if data is not loaded
        """
        if self.data is None:
            return None
            
        # Average alcohol content by quality rating
        avg_alcohol_by_quality = self.data.groupby('quality')['alcohol'].agg(['mean', 'std', 'count'])
        
        # Overall dataset statistics
        desc_stats = {
            'avg_alcohol_by_quality': avg_alcohol_by_quality,
            'correlation_matrix': self.data[self.feature_columns + ['quality']].corr(),
            'basic_stats': self.data.describe()
        }
        
        logger.info("Descriptive statistics calculated")
        return desc_stats
    
    def train_model(self):
        """
        Trains a Random Forest model on the processed wine quality data.
        
        This function:
        - Splits data into training and test sets
        - Scales features using StandardScaler
        - Trains a Random Forest model
        - Calculates performance metrics
        
        Returns:
            bool: True if model training was successful, False otherwise
            
        Note:
            Uses 80-20 train-test split with random_state=42
        """
        if self.data is None:
            return False
            
        try:
            # Prepare features and target
            X = self.data[self.feature_columns]
            y = self.data['quality']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest model
            self.model = RandomForestRegressor(n_estimators=1000, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)
            
            self.model_metrics = {
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred),
                'train_mse': mean_squared_error(y_train, train_pred),
                'test_mse': mean_squared_error(y_test, test_pred),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'test_mae': mean_absolute_error(y_test, test_pred)
            }
            
            logger.info(f"Model trained successfully. Trained R²: {self.model_metrics['train_r2']:.3f}")
            return True
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def predict_quality(self, fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                      chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                      ph, sulphates, alcohol):
        """
        Predicts wine quality based on all input features.
        
        Args:
            fixed_acidity (float): Fixed acidity (g/L)
            volatile_acidity (float): Volatile acidity (g/L)
            citric_acid (float): Citric acid (g/L)
            residual_sugar (float): Residual sugar (g/L)
            chlorides (float): Chlorides (g/L)
            free_sulfur_dioxide (float): Free sulfur dioxide (mg/L)
            total_sulfur_dioxide (float): Total sulfur dioxide (mg/L)
            ph (float): pH level
            sulphates (float): Sulphates (g/L)
            alcohol (float): Alcohol content (% by volume)
            
        Returns:
            dict or None: Dictionary containing:
                - prediction: Predicted quality score
                - feature_importance: Dictionary of feature importances
            Returns None if model is not trained
            
        Note:
            Features are automatically scaled before prediction
        """
        if self.model is None:
            return None
            
        try:
            # Create feature vector with all features
            features = np.array([[
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                ph, sulphates, alcohol
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            # Get feature importance for decision support
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            
            logger.info(f"Prediction made: {prediction:.2f}")
            return {
                'prediction': prediction,
                'feature_importance': feature_importance
            }
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def get_data_for_visualization(self):
        """
        Returns the processed dataset for visualization purposes.
        
        Returns:
            pd.DataFrame or None: The processed dataset if available, None otherwise
        """
        if self.data is None:
            return None
        return self.data
    
    def get_model_metrics(self):
        """
        Returns the model's performance metrics.
        
        Returns:
            dict or None: Dictionary containing:
                - train_r2: Training R² score
                - test_r2: Test R² score
                - train_mse: Training Mean Squared Error
                - test_mse: Test Mean Squared Error
                - train_mae: Training Mean Absolute Error
                - test_mae: Test Mean Absolute Error
            Returns None if model is not trained
        """
        return getattr(self, 'model_metrics', None) 