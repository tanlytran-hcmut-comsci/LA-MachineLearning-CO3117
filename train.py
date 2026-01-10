import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
import os
import pandas as pd
import time

from dataset import OCRDataset
from model import OCRModel


# Mapping for 47 classes
# 0-9: Digits
# 10-35: Uppercase
# 36-46: Lowercase (only specific ones)

label_map = {}
# Digits 0-9
for i in range(10):
    label_map[i] = str(i)
# Uppercase A-Z (Starts at 10)
for i in range(26):
    label_map[i + 10] = chr(65 + i)
# Lowercase (Starts at 36)
lowercase_chars = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
for i, char in enumerate(lowercase_chars):
    label_map[i + 36] = char

# Number of classes in EMNIST Balanced dataset
num_classes = 47

def get_char(label):
    return label_map.get(label)
#####

IMG_SIZE = (28, 28)

# def load_images(df, char_to_int):   # for image dataset
#     #### Load images from dataframe and convert to arrays
    
#     images = []
#     labels = []

#     for _, row in df.iterrows():
#         image_path = row['image_path']
#         label = row['label']

#         # convert all the labels into integers using the mapping we created
#         label_int = char_to_int[label]

#         # load the images and convert to array
#         img = load_img(image_path, target_size=IMG_SIZE, color_mode="grayscale")

#         img_array = img_to_array(img)

#         # normalize the array
#         img_array = img_array / 255.0

#         images.append(img_array)
#         labels.append(label_int)
    
#     return np.array(images), np.array(labels)


if __name__ == "__main__":
    # Load dataset
    print("Loading dataset...")
    dataset = OCRDataset()
    print(f"Total training samples: {len(dataset.X_train)}")
    
    # Load images
    print("Loading and preprocessing images...")
    X_train, y_train = dataset.X_train, dataset.y_train
    print("Raw shape:", X_train.shape)
    
    # reshape the images into the CNN input format [num_samples, height, width, channels]
    X_train = X_train.reshape(-1, 28, 28, 1)
    
    # one hot encode the labels
    y_train = to_categorical(y_train, num_classes)
    print("Reshaped:", X_train.shape)
    
    # reverse onehot encoding to split data while maintaining class distribution
    y_int_labels = np.argmax(y_train, axis=1)
    X_train, X_val, y_train_int, y_val_int = train_test_split(
        X_train, y_int_labels, test_size=0.2, stratify=y_int_labels, random_state=42
    )
    
    # convert back to onehot encoding
    y_train = to_categorical(y_train_int, num_classes)
    y_val = to_categorical(y_val_int, num_classes)
    
    # Verify class distribution
    print("Training labels distribution:", np.unique(y_train_int, return_counts=True))
    print("Validation labels distribution:", np.unique(y_val_int, return_counts=True))
    
    # Get data augmentation from dataset
    print("Setting up data augmentation----")
    data_generation = dataset.get_data_augmentation()
    
    # Create and compile model
    print("Building model----")
    ocr_model = OCRModel(num_classes=num_classes, input_shape=(28, 28, 1), learning_rate=0.001)
    ocr_model.compile_model()
    ocr_model.summary()
    
    # Setup callbacks for learning rate scheduling and early stopping
    print("Setting up training callbacks----")
    weights_dir = "./weight"
    stats_dir = "./epoch_statistic"
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    best_model_path = os.path.join(weights_dir, "best_ocr_model.h5")
    epoch_log_path = os.path.join(stats_dir, "training_log.csv")
    
    callbacks = [
        CSVLogger(
            epoch_log_path,
            separator=',',
            append=False
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,              # Reduce LR by half
            patience=3,              # Wait 3 epochs before reducing
            min_lr=1e-6,             # Don't go below this
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,              # Stop if no improvement for 7 epochs
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            best_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        )
    ]
    
    # Train model with more epochs
    print("\nStarting training with 50 epochs----")
    print("Note: Training will stop early if validation accuracy doesn't improve for 7 epochs")
    
    training_start_time = time.time()
    history = ocr_model.train(
        data_generation=data_generation,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=50,
        batch_size=32,
        callbacks=callbacks
    )
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    # Save the final model
    print("Saving final model----")
    model_path = os.path.join(weights_dir, "ocr_model.h5")
    ocr_model.save_model(model_path)
    
    print(f"\nTraining completed successfully!")
    print(f"\nTotal training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    print(f"Average time per epoch: {total_training_time/len(history.history['loss']):.2f} seconds")
    print(f"Best model saved at: {best_model_path}")
    print(f"Final model saved at: {model_path}")
    print(f"Epoch statistics saved at: {epoch_log_path}")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")


