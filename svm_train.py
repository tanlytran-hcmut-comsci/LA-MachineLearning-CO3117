from dataset import OCRDataset
from svm_model import SVMOCR
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
from sklearn.model_selection import GridSearchCV

# # --- OPTIMIZATION CONFIGURATION ---
# USE_HOG = False
# TYPE_KERNEL = 'rbf'  # 'linear' or 'rbf'

# PARAM_GRID = {
#     'n_components': [50, 150],  # PCA dimensions
#     'C': [1, 5, 10],                  # Regularization strength
# }
# # Total combinations: 3 * 3 = 9

USE_HOG = True
TYPE_KERNEL = 'rbf'  # 'linear' or 'rbf'

PARAM_GRID = {
    'n_components': [100],  # PCA dimensions
    'C': [ 5],                  # Regularization strength
}

CV_FOLDS = 5 

def main():
    # Setup directories
    stats_dir = "./epoch_statistic"
    weights_dir = "./weight"
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    
    log_path = os.path.join(stats_dir, "svm_training_log.csv")
    iterations_log_path = os.path.join(stats_dir, "svm_cv_iterations.csv")
    
    # Model path based on kernel
    model_path = os.path.join(weights_dir, f"emnist_svm_model_{'HOG' if USE_HOG else 'noHOG'}_{TYPE_KERNEL}_best.pkl")
    
    # Load Data
    print("Loading EMNIST Dataset ---")
    dataset = OCRDataset()
    print(f"Total Training Samples: {dataset.X_train.shape[0]:,}")
    print(f"Test Samples (held out): {dataset.X_test.shape[0]:,}")
    
    # Prepare data for optimization
    print("\nPreparing Data for Cross-Validation ---")
    print(f"Using ALL {dataset.X_train.shape[0]:,} training samples for {CV_FOLDS}-fold CV")
    
    # Setup GridSearchCV for Hyperparameter Optimization
    print("\n=== GridSearchCV Optimization ===")
    total_combinations = 1
    for values in PARAM_GRID.values():
        total_combinations *= len(values)
    print(f"Testing {total_combinations} combinations with {CV_FOLDS}-fold Cross Validation")
    print(f"Kernel: {TYPE_KERNEL}")
    print(f"HOG Features: {USE_HOG}")
    print("Parameter Search Space:")
    for param, values in PARAM_GRID.items():
        print(f"  {param}: {values}")
    print(f"\n Running SEQUENTIALLY (n_jobs=1) to prevent laptop crash")
    print(f"Estimated time: ~{total_combinations * CV_FOLDS * 1.5:.0f}-{total_combinations * CV_FOLDS * 3:.0f} minutes\n")
    
    # Use SVMOCR estimator (sklearn-compatible!)
    base_estimator = SVMOCR(
        kernel=TYPE_KERNEL,
        gamma='scale',
        use_hog=USE_HOG
    )
    
    # GridSearchCV - sequentially tests ALL combinations
    grid_search = GridSearchCV(
        estimator=base_estimator,
        param_grid=PARAM_GRID,
        cv=CV_FOLDS,
        verbose=2,
        n_jobs=2,  # Sequential 
        scoring='accuracy',
        return_train_score=True
    )
    
    # Run Optimization
    print("Starting Optimization ---")
    start_time = time.time()
    grid_search.fit(dataset.X_train, dataset.y_train)
    
    optimization_time = time.time() - start_time
    print(f"\n✅ Optimization completed in {optimization_time:.2f} seconds ({optimization_time/60:.2f} minutes)")
    
    # Log all iterations to CSV
    print("\nLogging All Combinations ---")
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    # Extract relevant columns
    iterations_log = pd.DataFrame({
        'n_components': cv_results['param_n_components'],
        'C': cv_results['param_C'],
        'avg_accuracy_all_folds': cv_results['mean_test_score'] * 100,
        'std_accuracy': cv_results['std_test_score'] * 100,
        'training_time_seconds': cv_results['mean_fit_time']
    })
    
    # Sort by accuracy (best first)
    iterations_log = iterations_log.sort_values('avg_accuracy_all_folds', ascending=False)
    
    # Save iterations log
    iterations_log.to_csv(iterations_log_path, index=False)
    print(f"All {len(iterations_log)} combinations logged to: {iterations_log_path}")
    
    # Get Best Parameters
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    print("\n=== BEST PARAMETERS FOUND ===")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best Cross-Validation Accuracy: {best_cv_score*100:.2f}%")
    
    # Get the best trained model
    print("\nExtracting Best Model ---")
    svm_model = grid_search.best_estimator_
    
    # Save model
    print("Saving Best Model ---")
    svm_model.save_model(model_path)
    
    # Log final best model statistics
    print("Logging Best Model Statistics ---")
    stats = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'kernel': TYPE_KERNEL,
        'use_hog': USE_HOG,
        'best_cv_accuracy': best_cv_score * 100,
        'optimization_time_minutes': optimization_time / 60,
        'total_combinations': total_combinations,
        'cv_folds': CV_FOLDS,
        'training_samples': dataset.X_train.shape[0],
        'model_path': model_path,
        **best_params
    }
    
    df = pd.DataFrame([stats])
    if os.path.exists(log_path):
        existing_df = pd.read_csv(log_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(log_path, index=False)
    
    print(f"Best model statistics saved to: {log_path}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Kernel:              {TYPE_KERNEL}")
    print(f"HOG Features:        {USE_HOG}")
    print(f"Best CV Accuracy:    {best_cv_score*100:.2f}%")
    print(f"Total Time:          {optimization_time/60:.2f} minutes")
    print(f"Total Combinations:  {total_combinations}")
    print(f"CV Folds:            {CV_FOLDS}")
    print(f"\nBest Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"\nModel saved to: {model_path}")
    print(f"All results saved to: {iterations_log_path}")
    print("="*60)

if __name__ == "__main__":
    main()