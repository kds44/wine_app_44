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
    # Class for processing wine quality data and making predictions
    
    def __init__(self):
        # Initializes the WineQualityProcessor with empty data and model
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
        # Loads the wine quality dataset from local CSV file
        try:
            # Load red wine dataset from local CSV file
            self.data = pd.read_csv("winequality-red.csv", sep=";")
            logger.info(f"Dataset loaded successfully: {self.data.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False

    def clean_and_feature_engineer(self):
        # Performs data cleaning and feature engineering on the wine dataset
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
        # Calculates and returns descriptive statistics for the dataset
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
        # Trains a Random Forest model on the processed wine quality data
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
        # Predicts wine quality based on all input features
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
        # Returns the processed dataset for visualization purposes
        if self.data is None:
            return None
        return self.data
    
    def get_model_metrics(self):
        # Returns the model's performance metrics
        return getattr(self, 'model_metrics', None) 