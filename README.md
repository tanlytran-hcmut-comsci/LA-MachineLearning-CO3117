# OCR Character Recognition - CNN vs Random Forest

A comprehensive Optical Character Recognition (OCR) system comparing CNN and Random Forest classifiers on the EMNIST Balanced dataset (47 classes: digits 0-9, uppercase A-Z, and 11 lowercase letters) (for CNN, 89,39% on test set, and 80,68% for Random Forest).

## Features

- **Two Model Architectures**: Deep Learning CNN and Random Forest classifier
- **EMNIST Balanced Dataset**: 47 character classes (112,800 training samples, 18,800 test samples)
- **Automated Hyperparameter Optimization**: GridSearchCV for Random Forest
- **Data Augmentation**: Rotation, Shifting, Zoom, and Linear Shear for CNN training
- **Performance Metrics**: Accuracy, inference time, training time tracking
- **Visualization**: Training history plots for CNN

## Requirements

Install dependencies:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn kagglehub joblib opencv-python
```

## Dataset Download

Download the EMNIST Balanced dataset using Kaggle API:

```bash
python download_data.py
```

This will download the dataset to: `~/.cache/kagglehub/datasets/crawford/emnist/versions/3/`
You have to change the dataset path in the "dataset.py" before using to retrain.

The dataset contains:
- `emnist-balanced-train.csv` - 112,800 training samples
- `emnist-balanced-test.csv` - 18,800 test samples

## Project Structure

```
.
├── dataset.py              # Dataset loader with preprocessing
├── download_data.py        # Script to download EMNIST dataset
│
├── model.py                # CNN model architecture
├── train.py                # Train CNN model
├── inference.py            # Test CNN model on test set
├── inference_image.py      # Predict single image with CNN
│
├── rf_model.py             # Random Forest model wrapper
├── rf_train.py             # Train Random Forest with GridSearchCV
├── rf_inference.py         # Test Random Forest on test set
│
├── visualize.py            # Training visualization utilities
│
├── weight/                 # Saved models
│   ├── best_ocr_model.h5          # Best CNN model
│   └── emnist_rf_model.pkl        # Best Random Forest model
│
└── epoch_statistic/        # Training logs and plots
    ├── training_log.csv           # CNN training history
    ├── rf_training_log.csv        # Random Forest best model stats
    └── rf_cv_iterations.csv       # All Random Forest combinations' configurations tested
```

## Training

### 1. Train CNN Model

Train the Convolutional Neural Network:

```bash
python train.py
```

**Configuration:**
- Architecture: 3 Block of Conv layers (32, 64, 128 filters) with Pooling + 2 Dense layers (256, 47)
- Optimizer: Adam (lr=0.001 with ReduceLROnPlateau)
- Epochs: 50 (with EarlyStopping patience=7)
- Batch size: 32
- Data augmentation: rotation, shift, zoom, shear

**Outputs:**
- `weight/best_ocr_model.h5` - Best model based on validation accuracy
- `weight/ocr_model.h5` - Final model
- `epoch_statistic/training_log.csv` - Per-epoch training metrics
- `epoch_statistic/training_accuracy.png` - Accuracy plot
- `epoch_statistic/training_loss.png` - Loss plot

### 2. Train Random Forest Model

Train Random Forest with exhaustive hyperparameter search:

```bash
python rf_train.py
```

**Configuration:**
- GridSearchCV with 5-fold cross-validation
- Parameter grid:
  - `n_estimators`: [100, 200]
  - `max_depth`: [10, 20, 30]
  - `min_samples_leaf`: [2, 4]
- Total combinations tested: 12

**Outputs:**
- `weight/emnist_rf_model.pkl` - Best Random Forest model
- `epoch_statistic/rf_training_log.csv` - Best model summary
- `epoch_statistic/rf_cv_iterations.csv` - All tested configurations with accuracy

## Inference

### CNN Inference on Test Set

Evaluate CNN on the entire test set (18,800 images):

```bash
python inference.py
```

**Output:**
- Test accuracy percentage
- Inference time and time per sample
- Sample predictions with confidence scores
- Error analysis

### Random Forest Inference on Test Set

Evaluate Random Forest on the entire test set:

```bash
python rf_inference.py
```

**Output:**
- Test accuracy percentage
- Inference time and time per sample
- Sample predictions with character labels and confidence scores

### Predict Single Image (CNN only)

Predict character from a custom image:

```bash
python inference_image.py path/to/your/image.png
```

## Visualization

Generate training history plots for CNN:

```bash
python visualize.py
```

Creates professional plots in `epoch_statistic/`:
- `training_accuracy.png` / `.pdf`
- `training_loss.png` / `.pdf`

## Expected Results

### CNN Model
- **Training time**: ~20-30 minutes (50 epochs with early stopping)
- **Test accuracy**: ~88-92%
- **Inference time**: ~1-2 seconds for 18,800 samples (~0.1 ms/sample)

### Random Forest Model
- **Training time**: ~60-90 minutes (12 configurations × 5 folds)
- **Test accuracy**: ~80-84%
- **Inference time**: ~5-10 seconds for 18,800 samples (~0.5 ms/sample)

## EMNIST Character Mapping

- Labels 0-9: Digits '0'-'9'
- Labels 10-35: Uppercase 'A'-'Z'
- Labels 36-46: Lowercase 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'

## Notes

- The EMNIST CSV format requires image rotation/flip correction (handled in `dataset.py`)
- CNN uses data augmentation to improve generalization
- Random Forest uses GridSearchCV for optimal hyperparameter selection
- All models use the same train/test split from the EMNIST dataset

## License

This project is for educational purposes.

## Acknowledgments

- EMNIST Dataset (CSV Style): [published in "EMNIST: Extending MNIST to handwritten letters" from Cohen et al.](https://arxiv.org/abs/1702.05373)
- EMNIST Kaggle Dataset (Image style): [crawford/emnist](https://www.kaggle.com/datasets/crawford/emnist)
