import numpy as np
import joblib
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import time
import os

class SVMOCR(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', use_hog=True, n_components=100):
        """
        SVM Wrapper for OCR with HOG feature extraction
        
        Args:
            C: Regularization parameter (default=1.0)
            kernel: Kernel type (default='rbf', best for handwriting)
            gamma: Kernel coefficient (default='scale')
            use_hog: Whether to use HOG features instead of raw pixels (default=True, recommended)
            n_components: Number of PCA components (default=100, None to disable PCA)
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.use_hog = use_hog
        self.n_components = n_components
        
        # HOG parameters 
        self.hog_orientations = 9
        self.hog_pixels_per_cell = (4, 4)
        self.hog_cells_per_block = (2, 2)
        
        # PCA will be initialized in fit() to ensure GridSearchCV parameter changes work
        self.pca = None
        
        # SVM is highly sensitive to scale, so we always include a scaler
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        self.training_stats = {}

    def _extract_hog_features(self, X):
        if len(X.shape) == 4:
            X = X.reshape(X.shape[0], 28, 28)
        elif len(X.shape) == 2:
            n_samples = X.shape[0]
            X = X.reshape(n_samples, 28, 28)
        
        hog_features = []
        for img in X:
            features = hog(
                img,
                orientations=self.hog_orientations,
                pixels_per_cell=self.hog_pixels_per_cell,
                cells_per_block=self.hog_cells_per_block,
                block_norm='L2-Hys',
                transform_sqrt=True,
                feature_vector=True
            )
            hog_features.append(features)
            print(f"HOG feature length: {len(features)}")
        return np.array(hog_features)

    def _flatten(self, X):
        """Convert images to feature vectors (HOG or raw pixels)"""
        if self.use_hog:
            return self._extract_hog_features(X)
        else:
            # Convert (N, 28, 28, 1) -> (N, 784)
            if len(X.shape) > 2:
                return X.reshape(X.shape[0], -1)
            return X

    def _prepare_labels(self, y):
        """Convert One-Hot (N, 47) -> Integers (N,)"""
        if len(y.shape) > 1 and y.shape[1] > 1:
            return np.argmax(y, axis=1)
        return y

    def fit(self, X, y):
        """Fit method compatible with sklearn's API"""
        print("Initializing SVM model...")
        start_fit_time = time.time()
        
        # Create SVM model - use LinearSVC for linear kernel (much faster)
        if self.kernel == 'linear':
            print("Using LinearSVC (optimized for linear kernel)...")
            self.model = LinearSVC(
                C=self.C,
                max_iter=2000,
                random_state=42,
                verbose=1,
                dual='gamma' 
            )
        else:
            print(f"Using SVC with {self.kernel} kernel...")
            self.model = SVC(
                C=self.C,
                kernel=self.kernel,
                max_iter=2000,
                gamma=self.gamma,
                cache_size=2048,
                decision_function_shape='ovo',  
                random_state=42,
                verbose=True
            )
        
        # Extract features and prepare labels
        print("Extracting features...")
        feature_start = time.time()
        X_features = self._flatten(X)
        y_int = self._prepare_labels(y)
        feature_time = time.time() - feature_start
        print(f"Feature extraction completed in {feature_time:.2f} seconds")
        print(f"Original feature dimensions: {X_features.shape[1]}")
        
        # Initialize PCA here to ensure GridSearchCV parameter changes work
        if self.n_components is not None:
            self.pca = PCA(n_components=self.n_components)
            print(f"Applying PCA to reduce dimensions to {self.n_components}...")
            pca_start = time.time()
            X_features = self.pca.fit_transform(X_features)
            pca_time = time.time() - pca_start
            print(f"PCA completed in {pca_time:.2f} seconds")
            print(f"Reduced feature dimensions: {X_features.shape[1]}")
            print(f"Explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")
        else:
            self.pca = None
        
        # Fit scaler and transform
        print("Scaling features")
        scale_start = time.time()
        X_scaled = self.scaler.fit_transform(X_features)
        scale_time = time.time() - scale_start
        print(f"Scaling completed in {scale_time:.2f} seconds")
        
        # Train SVM
        print("Training SVM classifier")
        train_start = time.time()
        self.model.fit(X_scaled, y_int)
        train_time = time.time() - train_start
        print(f"SVM training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
        
        total_time = time.time() - start_fit_time
        print(f"Total fit time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        self.is_trained = True
        
        return self

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train SVM with optional validation (similar to Random Forest format)"""
        print("Starting SVM Training ---")
        print(f"Training samples: {len(X_train):,}")
        
        feature_type = "HOG features" if self.use_hog else "raw pixels"
        
        # Extract features from first sample to get feature count
        sample_features = self._flatten(X_train[:1])
        print(f"Features per sample: {sample_features.shape[1]} ({feature_type})")
        
        # Train
        start_time = time.time()
        self.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Training accuracy
        train_preds = self.predict(X_train)
        y_train_int = self._prepare_labels(y_train)
        train_acc = accuracy_score(y_train_int, train_preds) * 100
        print(f"Training Accuracy: {train_acc:.2f}%")
        
        # Validation accuracy if provided
        if X_val is not None and y_val is not None:
            val_preds = self.predict(X_val)
            y_val_int = self._prepare_labels(y_val)
            val_acc = accuracy_score(y_val_int, val_preds) * 100
            print(f"Validation Accuracy: {val_acc:.2f}%")
            
            self.training_stats = {
                'training_time': training_time,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc
            }
        else:
            self.training_stats = {
                'training_time': training_time,
                'train_accuracy': train_acc
            }
        
        print("Training complete.")
        return self.training_stats

    def predict(self, X):
        """Predict class labels"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        X_features = self._flatten(X)
        if self.pca is not None:
            X_features = self.pca.transform(X_features)
        X_scaled = self.scaler.transform(X_features)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict class probabilities (requires SVM with probability=True)"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        # Note: SVC needs probability=True to use this, but it's slower
        # For now, return one-hot encoded predictions
        predictions = self.predict(X)
        n_classes = len(np.unique(predictions))
        probs = np.zeros((len(predictions), max(47, n_classes)))  # EMNIST has 47 classes
        probs[np.arange(len(predictions)), predictions] = 1.0
        return probs

    def save_model(self, filepath):
        """Save model with HOG configuration"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'n_components': self.n_components,
            'use_hog': self.use_hog,
            'hog_orientations': self.hog_orientations,
            'hog_pixels_per_cell': self.hog_pixels_per_cell,
            'hog_cells_per_block': self.hog_cells_per_block,
            'training_stats': self.training_stats
        }, filepath)
        print(f"SVM model saved to {filepath}")

    def load_model(self, filepath):
        """Load model and restore HOG configuration"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.pca = data.get('pca', None)
        self.n_components = data.get('n_components', None)
        
        # Restore HOG configuration if present
        self.use_hog = data.get('use_hog', True)
        self.hog_orientations = data.get('hog_orientations', 9)
        self.hog_pixels_per_cell = data.get('hog_pixels_per_cell', (4, 4))
        self.hog_cells_per_block = data.get('hog_cells_per_block', (2, 2))
        self.training_stats = data.get('training_stats', {})
        
        self.is_trained = True
        print(f"SVM model loaded from {filepath}")