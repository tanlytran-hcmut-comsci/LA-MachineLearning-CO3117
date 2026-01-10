import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np

class OCRModel:
    def __init__(self, num_classes, input_shape=(28, 28, 1), learning_rate=0.001):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        model = Sequential([
            # conv layer
            Conv2D(
                32,                             # number of filters
                (3, 3),                         # kernel_size
                activation='relu',              # activation function.
                kernel_regularizer=l2(0.001),   # lamda: regularization.
                input_shape=self.input_shape,   # input shape (image after preprocessing)
                padding='same'                  # keep the input size
            ),
            BatchNormalization(),  # keeps the network stable and helps it train faster by normalizing 
            # activations, then letting the model re-scale and re-shift them with learnable parameters.
            MaxPooling2D((2, 2)),
            Dropout(0.3),

            # conv layer
            Conv2D(64, (3, 3), 
                activation='relu', 
                kernel_regularizer=l2(0.001), 
                padding='same'
            ),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),

            # conv: captures high-level features
            Conv2D(128, (3, 3), 
                activation='relu', 
                padding='same', 
                kernel_regularizer=l2(0.001)
            ),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # fully connected layer: classification
            Flatten(),
            Dense(256, 
                activation='relu', 
                kernel_regularizer=l2(0.001)
            ),
            BatchNormalization(), # faster convergence
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')  # Output layer
        ])
        
        self.model = model
        return model
    
    def compile_model(self):
        if self.model is None:
            self.build_model()
            
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        
    def summary(self):
        # display model architecture
        if self.model is None:
            self.build_model()
        self.model.summary()
        
    def train(self, data_generation, X_train, y_train, 
                    X_val, y_val, epochs=20, batch_size=32, callbacks=None):
        if self.model is None:
            self.compile_model()
            
        data_generation.fit(X_train)
        
        self.history = self.model.fit(
            data_generation.flow(X_train, y_train, batch_size=batch_size), 
            epochs=epochs, 
            validation_data=(X_val, y_val),
            callbacks=callbacks if callbacks else []
        )
        
        return self.history
    
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("We do not have model to save =)))")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
    def predict(self, X):
        if self.model is None:
            raise ValueError("We do not have model to predict =)))")
        return self.model.predict(X)
