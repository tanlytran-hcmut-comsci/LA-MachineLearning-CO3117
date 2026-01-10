from dataset import OCRDataset
from rf_model import RandomForestOCR
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
from sklearn.model_selection import GridSearchCV

# --- OPTIMIZATION CONFIGURATION ---
PARAM_GRID = {
    'n_estimators': [100, 200],        # Number of trees
    'max_depth': [10, 20, 30],    # How deep each tree can grow
    'min_samples_leaf': [2, 4],          # Minimum samples allowed in a leaf
}
# Total combinations: 2 * 3 * 2 = 12

CV_FOLDS = 5 # 5-fold cross-validation (20/80 split per fold)

def main():
    # Setup directories
    stats_dir = "./epoch_statistic"
    weights_dir = "./weight"
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    
    log_path = os.path.join(stats_dir, "rf_training_log.csv")
    model_path = os.path.join(weights_dir, "emnist_rf_model.pkl")
    
    # Load Data
    print("Loading EMNIST Dataset ---")
    dataset = OCRDataset() 
    print(f"Total Training Samples: {dataset.X_train.shape[0]:,}")
    print(f"Test Samples (held out): {dataset.X_test.shape[0]:,}")
    
    # Prepare ALL training data for optimization
    print("Preparing Data for Cross-Validation ---")
    print(f"Using ALL {dataset.X_train.shape[0]:,} training samples for {CV_FOLDS}-fold CV")
    print(f"Each fold: {int(dataset.X_train.shape[0] * 0.8):,} train, {int(dataset.X_train.shape[0] * 0.2):,} validation")
    
    # Setup GridSearchCV for Hyperparameter Optimization
    print("GridSearchCV Optimization ---")
    total_combinations = 1
    for values in PARAM_GRID.values():
        total_combinations *= len(values)
    print(f"Testing ALL {total_combinations} combinations with {CV_FOLDS}-fold Cross Validation...")
    print("Parameter Search Space:")
    for param, values in PARAM_GRID.items():
        print(f"  {param}: {values}")
    
    # Use RandomForestOCR estimator (now sklearn-compatible!)
    base_estimator = RandomForestOCR(n_jobs=2)
    
    # GridSearchCV - exhaustively tests ALL combinations
    grid_search = GridSearchCV(
        estimator=base_estimator,
        param_grid=PARAM_GRID,
        cv=CV_FOLDS,
        verbose=2,
        n_jobs=2,
        scoring='accuracy',
        return_train_score=True  # Log training
    )
    
    # Run Optimization
    print("Starting Optimization ---")
    start_time = time.time()
    grid_search.fit(dataset.X_train, dataset.y_train)
    
    optimization_time = time.time() - start_time
    print(f"Optimization completed in {optimization_time:.2f} seconds ({optimization_time/60:.2f} minutes)")
    
    # Log all iterations to CSV
    print("Logging All Combinations ---")
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    # Extract relevant columns - parameters and average accuracy across folds
    iterations_log = pd.DataFrame({
        'n_estimators': cv_results['param_n_estimators'],
        'max_depth': cv_results['param_max_depth'],
        'min_samples_leaf': cv_results['param_min_samples_leaf'],
        'avg_accuracy_all_folds': cv_results['mean_test_score'] * 100,  # Convert to percentage
        'training_time_seconds': cv_results['mean_fit_time']
    })
    
    # Sort by accuracy (best first)
    iterations_log = iterations_log.sort_values('avg_accuracy_all_folds', ascending=False)
    
    # Save iterations log
    iterations_log_path = os.path.join(stats_dir, "rf_cv_iterations.csv")
    iterations_log.to_csv(iterations_log_path, index=False)
    print(f"All {len(iterations_log)} combinations logged to: {iterations_log_path}")
    
    # Get Best Parameters
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    print("\n--- BEST PARAMETERS FOUND ---")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best Cross-Validation Accuracy: {best_cv_score*100:.2f}%")
    
    # Get the best trained model (already a RandomForestOCR instance!)
    print("Get Best Model ---")
    rf_model = grid_search.best_estimator_
    
    total_time = time.time() - start_time
    
    # Save model
    print("Saving Best Model ---")
    rf_model.save_model(model_path)

    
    # Log final best model statistics to CSV
    print("Logging Best Model Statistics ---")
    stats = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_cv_accuracy': best_cv_score,
        'optimization_time': optimization_time,
        'total_time': total_time,
        'total_combinations': total_combinations,
        'cv_folds': CV_FOLDS,
        'training_samples': dataset.X_train.shape[0],
        'model_path': model_path,
        **best_params  # Include all best parameters as columns
    }
    
    df = pd.DataFrame([stats])
    if os.path.exists(log_path):
        existing_df = pd.read_csv(log_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(log_path, index=False)
    
    print(f"Best model statistics saved to: {log_path}")
    print(f"All combinations saved to: {iterations_log_path}")
    
    # Summary
    print("TRAINING SUMMARY: ")
    print(f"Best CV Accuracy:    {best_cv_score*100:.2f}%")
    print(f"Total Time:          {total_time/60:.2f} minutes")
    print(f"Total Combinations:  {total_combinations}")
    print(f"CV Folds:            {CV_FOLDS} (20/80 split per fold)")
    print(f"\nBest Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

if __name__ == "__main__":
    main()