import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import os
import time

class RandomForestOCR(BaseEstimator, ClassifierMixin): 
    ### Compatible with sklearn's RandomizedSearchCV for hyperparameter optimization.
    
    def __init__(self, n_estimators=200, max_depth=30, min_samples_split=10, 
                 min_samples_leaf=2, max_features='sqrt', class_weight='balanced',
                 n_jobs=-1, use_scaler=False):
        """
        Initialize Random Forest with optimized hyperparameters for EMNIST
        
        Args:
            n_estimators: Number of trees (200 is good balance of accuracy vs speed)
            max_depth: Maximum tree depth (30 prevents overfitting on 28x28 images)
            min_samples_split: Minimum samples to split a node (reduces overfitting)
            min_samples_leaf: Minimum samples in leaf node (smooths predictions)
            max_features: Features to consider for split ('sqrt' is standard for classification)
            class_weight: Handle class imbalance ('balanced' or None)
            n_jobs: CPU cores to use (-1 = all cores)
            use_scaler: Whether to standardize features (usually not needed for images)
        """
        # Store all parameters as instance attributes (required by sklearn's BaseEstimator)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.use_scaler = use_scaler
        
        # Don't create model here - create it in fit() so RandomizedSearchCV can update params
        self.model = None
        self.scaler = StandardScaler() if self.use_scaler else None
        self.is_trained = False
        self.training_stats = {}
        self.feature_importance = None

    def _flatten(self, X):
        # Converts (N, 28, 28, 1) to (N, 784) since sklearn requires 2D 
        # arrays (samples, features), not 4D as CNN

        if len(X.shape) == 2:
            return X
        return X.reshape(X.shape[0], -1)

    def _prepare_labels(self, y):
        # Converts One-Hot (N, 47) -> Integers (N,) since sklearn prefers 
        # class indices but not one-hot vectors.
        if len(y.shape) > 1 and y.shape[1] > 1:
            return np.argmax(y, axis=1)
        return y

    def fit(self, X, y):
        ### Fit method compatible with sklearn's API (used by RandomizedSearchCV)
        # Create model with current parameters (allows RandomizedSearchCV to update params)
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=42,
            verbose=1,
            class_weight=self.class_weight
        )
        
        X_flat = self._flatten(X)
        y_int = self._prepare_labels(y)
        
        if self.scaler:
            X_flat = self.scaler.fit_transform(X_flat)
        
        self.model.fit(X_flat, y_int)
        self.is_trained = True
        self.feature_importance = self.model.feature_importances_
        
        return self  # sklearn convention: fit() returns self

    def train(self, X_train, y_train, X_val=None, y_val=None):
        print(f"Preprocessing data: (Flattening images)")
        X_flat = self._flatten(X_train)
        y_int = self._prepare_labels(y_train)
        
        # Apply scaling if enabled
        if self.scaler:
            X_flat = self.scaler.fit_transform(X_flat)
        
        print(f"Training samples: {X_flat.shape[0]:,}")
        print(f"Features per sample: {X_flat.shape[1]:,}")
        print(f"Number of classes: {len(np.unique(y_int))}")
        
        # Check class distribution
        unique, counts = np.unique(y_int, return_counts=True)
        print(f"Class distribution: Min={counts.min()}, Max={counts.max()}, Avg={counts.mean():.0f}")
        
        # Train model
        print(f"Training {self.model.n_estimators} decision trees.")
        start_time = time.time()
        
        self.model.fit(X_flat, y_int)
        
        train_time = time.time() - start_time
        self.is_trained = True
        
        print(f"Training completed in {train_time:.2f} seconds")
        
        # Calculate feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Evaluate on training set
        train_pred = self.model.predict(X_flat)
        train_acc = accuracy_score(y_int, train_pred)
        
        print(f"Training Accuracy: {train_acc*100:.2f}%")
        
        # Store training statistics
        self.training_stats = {
            'train_accuracy': train_acc,
            'train_time': train_time,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'n_samples': X_flat.shape[0],
            'n_features': X_flat.shape[1],
            'n_classes': len(unique)
        }
        
        # Validation evaluation
        if X_val is not None and y_val is not None:
            val_acc = self.evaluate(X_val, y_val)
            self.training_stats['val_accuracy'] = val_acc
            print(f"Validation Accuracy: {val_acc*100:.2f}%")
        

    def evaluate(self, X_test, y_test):
        """Evaluate model accuracy on test set"""
        X_flat = self._flatten(X_test)
        y_int = self._prepare_labels(y_test)
        
        if self.scaler:
            X_flat = self.scaler.transform(X_flat)
        
        predictions = self.model.predict(X_flat)
        accuracy = accuracy_score(y_int, predictions)
        
        return accuracy

    def predict(self, X):
        ### Returns class predictions (sklearn convention)
        if not self.is_trained:
            raise ValueError("Model is not trained =)))")
            
        X_flat = self._flatten(X)
        
        if self.scaler:
            X_flat = self.scaler.transform(X_flat)
        
        return self.model.predict(X_flat)
    
    def predict_proba(self, X):
        ### Returns probabilities (N, 47) to match CNN output format.
        if not self.is_trained:
            raise ValueError("Model is not trained =)))")
            
        X_flat = self._flatten(X)
        
        if self.scaler:
            X_flat = self.scaler.transform(X_flat)
        
        return self.model.predict_proba(X_flat)

    def save_model(self, filepath):
        ### Save model and scaler to disk
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        print(f"Random Forest model saved to {filepath}")

    def load_model(self, filepath):
        ### Load model and scaler from disk
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data.get('scaler', None)
        self.feature_importance = model_data.get('feature_importance', None)
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")