import os
import numpy as np
import time

from dataset import OCRDataset
from model import OCRModel

# Mapping for 47 classes (same as train.py)
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
    return label_map.get(label, '?')


def evaluate_predictions(true_labels, predicted_labels):
    """Calculate accuracy from integer labels"""
    correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    accuracy = correct / len(true_labels) * 100
    return accuracy, correct


if __name__ == "__main__":
    # Load dataset
    print("Loading EMNIST test dataset...")
    dataset = OCRDataset()
    X_test = dataset.X_test
    y_test = dataset.y_test
    print(f"Total test samples: {len(X_test)}\n")
    print(f"Test data shape: {X_test.shape}")
    
    # Load the trained model
    print("\nLoading trained model...")
    model_path = "./weight/best_ocr_model.h5"  # Use best model
    
    if not os.path.exists(model_path):
        print(f"Best model not found, trying final model...")
        model_path = "./weight/best_ocr_model.h5"
        if not os.path.exists(model_path):
            print(f"Error: Model file not found")
            print("Please train the model first by running: python train.py")
            exit(1)
    
    ocr_model = OCRModel(num_classes=num_classes, input_shape=(28, 28, 1))
    ocr_model.load_model(model_path)
    
    # Make predictions
    print(f"\nMaking predictions on {len(X_test)} images...")
    
    start_time = time.time()
    y_pred = ocr_model.predict(X_test)
    inference_time = time.time() - start_time
    
    # Get the predicted class labels
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    print(f"Predictions completed for all {len(y_pred_labels)} images!")
    print(f"Inference time:       {inference_time:.2f} seconds")
    print(f"Time per sample:      {(inference_time/len(X_test))*1000:.2f} ms")
    
    # Calculate accuracy
    accuracy, correct = evaluate_predictions(y_test, y_pred_labels)
    print(f"\nTest Accuracy: {accuracy:.2f}% ({correct}/{len(y_test)} correct)")
    
    # Show sample predictions
    print("\nSample predictions (first 20):")
    print("-" * 80)
    for i in range(min(20, len(y_pred_labels))):
        true_char = get_char(y_test[i])
        pred_char = get_char(y_pred_labels[i])
        status = "✓" if y_test[i] == y_pred_labels[i] else "✗"
        confidence = y_pred[i][y_pred_labels[i]] * 100
        print(f"{status} True: {true_char:3s} | Predicted: {pred_char:3s} | Confidence: {confidence:.1f}%")
    
    # Error analysis
    print("\nError Analysis:")
    errors = [(y_test[i], y_pred_labels[i]) for i in range(len(y_test)) if y_test[i] != y_pred_labels[i]]
    if errors:
        print(f"Total errors: {len(errors)}")
        print("\nMost common misclassifications (first 10):")
        for i, (true_label, pred_label) in enumerate(errors[:10]):
            true_char = get_char(true_label)
            pred_char = get_char(pred_label)
            print(f"  {i+1}. True: {true_char} - Predicted: {pred_char}")
    else:
        print("Perfect predictions! No errors found.")
    
    print("\nInference completed successfully!")
    