from svm_model import SVMOCR
from dataset import OCRDataset
import numpy as np
import time
import os

# EMNIST Balanced mapping (47 classes)
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

def get_emnist_char(label):
    """Convert EMNIST numeric label to character"""
    return label_map.get(label, '?')

def main():
    model_path = "./weight/emnist_svm_model_HOG_rbf_best.pkl"
    
    # 1. Load model
    print("Load Model ---")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        print("Please train the model first by running: python svm_train.py")
        return
    
    svm_model = SVMOCR()
    svm_model.load_model(model_path)
    
    # 2. Load test data
    print("Load Test Data ---")
    dataset = OCRDataset()
    X_test = dataset.X_test
    y_test = dataset.y_test
    print(f"Test samples: {len(X_test):,}")
    
    # 3. Run inference
    print("Run Inference ---")
    start_time = time.time()
    
    # Predict
    pred_classes = svm_model.predict(X_test)
    
    inference_time = time.time() - start_time
    
    # Convert true labels (handle one-hot encoding)
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        true_classes = np.argmax(y_test, axis=1)
    else:
        true_classes = y_test
    
    # Calculate accuracy
    correct = np.sum(pred_classes == true_classes)
    accuracy = (correct / len(true_classes)) * 100
    
    # 4. Display results
    print("INFERENCE RESULTS: ")
    print(f"Total samples:        {len(true_classes):,}")
    print(f"Correct predictions:  {correct:,}")
    print(f"Accuracy:             {accuracy:.2f}%")
    print(f"Inference time:       {inference_time:.2f} seconds")
    print(f"Time per sample:      {(inference_time/len(true_classes))*1000:.2f} ms")
    
    # Show sample predictions
    print("\nSample Predictions (First 10):")
    print("-"*70)
    print(f"{'True':<10} {'Predicted':<12} {'Status'}")
    print("-"*70)
    for i in range(min(10, len(pred_classes))):
        status = "✓" if true_classes[i] == pred_classes[i] else "✗"
        true_char = get_emnist_char(true_classes[i])
        pred_char = get_emnist_char(pred_classes[i])
        print(f"{true_char} ({true_classes[i]:<2})   {pred_char} ({pred_classes[i]:<2})      {status}")
    
    print("\nInference Complete.")

if __name__ == "__main__":
    main()