import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

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


def preprocess_image(image_path, target_size=(28, 28), show_preview=False):
    """
    Load and preprocess an image for OCR prediction
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (28, 28)
        show_preview: Whether to show preprocessing steps
        
    Returns:
        Preprocessed image array ready for prediction
    """
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize to target size
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize to 0-1 range
    img_normalized = img_resized / 255.0
    
    # Add channel dimension: (28, 28) -> (28, 28, 1)
    img_processed = img_normalized.reshape(1, 28, 28, 1)
    
    if show_preview:
        # Show preprocessing steps
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(img_resized, cmap='gray')
        axes[1].set_title(f'Resized to {target_size}')
        axes[1].axis('off')
        
        axes[2].imshow(img_normalized, cmap='gray')
        axes[2].set_title('Normalized (0-1)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return img_processed


def predict_image(model, image_path, show_preview=False):
    """
    Predict character from an image file
    
    Args:
        model: Loaded OCRModel instance
        image_path: Path to the image file
        show_preview: Whether to show preprocessing steps
        
    Returns:
        predicted_char: Predicted character
        confidence: Prediction confidence (0-100)
    """
    # Preprocess image
    img_processed = preprocess_image(image_path, show_preview=show_preview)
    
    # Make prediction
    predictions = model.predict(img_processed, verbose=0)
    
    # Get predicted class and confidence
    predicted_label = np.argmax(predictions[0])
    confidence = predictions[0][predicted_label] * 100
    
    # Convert to character
    predicted_char = get_char(predicted_label)
    
    return predicted_char, confidence, predictions[0]


def predict_batch_images(model, image_paths, show_results=True):
    """
    Predict characters from multiple image files
    
    Args:
        model: Loaded OCRModel instance
        image_paths: List of image file paths
        show_results: Whether to display results
        
    Returns:
        results: List of (image_path, predicted_char, confidence) tuples
    """
    results = []
    
    for image_path in image_paths:
        try:
            predicted_char, confidence, _ = predict_image(model, image_path)
            results.append((image_path, predicted_char, confidence))
            
            if show_results:
                print(f"Image: {os.path.basename(image_path):30s} → Predicted: {predicted_char:3s} (Confidence: {confidence:.1f}%)")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append((image_path, None, 0))
    
    return results


if __name__ == "__main__":
    # Load the trained model
    print("Loading trained model...")
    model_path = "./weight/best_ocr_model.h5"
    
    if not os.path.exists(model_path):
        print(f"Best model not found, trying final model...")
        model_path = "./weight/ocr_model.h5"
        if not os.path.exists(model_path):
            print(f"Error: Model file not found")
            print("Please train the model first by running: python train.py")
            exit(1)
    
    ocr_model = OCRModel(num_classes=num_classes, input_shape=(28, 28, 1))
    ocr_model.load_model(model_path)
    print(f"Model loaded from {model_path}\n")
    
    # Check if image path provided as command line argument
    if len(sys.argv) > 1:
        # Single or multiple images from command line
        image_paths = sys.argv[1:]
        
        if len(image_paths) == 1:
            # Single image prediction with preview
            image_path = image_paths[0]
            print(f"Predicting character from: {image_path}\n")
            
            try:
                predicted_char, confidence, probabilities = predict_image(
                    ocr_model, image_path, show_preview=True
                )
                
                print(f"\n{'='*60}")
                print(f"Predicted Character: {predicted_char}")
                print(f"Confidence: {confidence:.2f}%")
                print(f"{'='*60}")
                
                # Show top 5 predictions
                print("\nTop 5 predictions:")
                top_5_indices = np.argsort(probabilities)[-5:][::-1]
                for i, idx in enumerate(top_5_indices, 1):
                    char = get_char(idx)
                    prob = probabilities[idx] * 100
                    print(f"  {i}. {char:3s} - {prob:.2f}%")
                    
            except Exception as e:
                print(f"Error: {e}")
                exit(1)
        else:
            # Multiple images prediction
            print(f"Predicting characters from {len(image_paths)} images...\n")
            results = predict_batch_images(ocr_model, image_paths)
            
            # Summary
            successful = sum(1 for _, char, _ in results if char is not None)
            print(f"\nProcessed {successful}/{len(image_paths)} images successfully")
    else:
        # Interactive mode
        print("OCR Image Inference - Interactive Mode")
        print("="*60)
        print("\nUsage:")
        print("  python inference_image.py <image_path>           - Predict single image")
        print("  python inference_image.py <img1> <img2> ...      - Predict multiple images")
        print("\nSupported formats: PNG, JPG, JPEG")
        print("\nExample:")
        print("  python inference_image.py my_character.png")
        print("="*60)
        
        # Prompt for image path
        while True:
            image_path = input("\nEnter image path (or 'quit' to exit): ").strip()
            
            if image_path.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not image_path:
                continue
            
            try:
                predicted_char, confidence, probabilities = predict_image(
                    ocr_model, image_path, show_preview=True
                )
                
                print(f"\n{'='*60}")
                print(f"Predicted Character: {predicted_char}")
                print(f"Confidence: {confidence:.2f}%")
                print(f"{'='*60}")
                
                # Show top 5 predictions
                print("\nTop 5 predictions:")
                top_5_indices = np.argsort(probabilities)[-5:][::-1]
                for i, idx in enumerate(top_5_indices, 1):
                    char = get_char(idx)
                    prob = probabilities[idx] * 100
                    print(f"  {i}. {char:3s} - {prob:.2f}%")
                    
            except Exception as e:
                print(f"Error: {e}")