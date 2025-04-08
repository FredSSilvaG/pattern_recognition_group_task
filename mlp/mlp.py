from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from file_utils import get_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import time
import pandas as pd
import os


# Function to log experiment results
def log_experiment(results, hidden_layer_sizes, learning_rate, val_accuracy, final_loss, run_time):
    results.append({
        'hidden_layer_sizes': hidden_layer_sizes,
        'learning_rate': learning_rate,
        'validation_accuracy': val_accuracy,
        'final_loss': final_loss,
        'run_time': run_time
    })
    return results


# Hyperparameter tuning function
def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    hidden_layer_sizes_options = [
        (64,),  # Single layer
        (128,),  # Single layer
        (256,),  # Single layer
        (128, 64),  # Two layers
        (256, 128),  # Two layers
        (256, 128, 64)  # Three layers
    ]
    learning_rates = [0.0001, 0.001, 0.01, 0.1]

    best_accuracy = 0
    best_params = {}
    results = []  # To store all experiment results

    for hidden_layer_sizes in hidden_layer_sizes_options:
        for lr in learning_rates:
            print(f'Training with hidden_layer_sizes={hidden_layer_sizes}, learning_rate={lr}')
            start_time = time.time()  # Record start time

            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                learning_rate_init=lr,
                max_iter=500,  # Increase iterations
                random_state=42
            )
            model.fit(X_train, y_train)

            # Get final loss and accuracy
            final_loss = model.loss_
            val_accuracy = model.score(X_val, y_val)
            run_time = time.time() - start_time  # Calculate run time

            # Log experiment results
            results = log_experiment(results, hidden_layer_sizes, lr, val_accuracy, final_loss, run_time)

            print(f'Final Loss: {final_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Run Time: {run_time:.2f}s')

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_params = {'hidden_layer_sizes': hidden_layer_sizes, 'learning_rate': lr}

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df, best_params, best_accuracy


def report_result(file_path, best_accuracy, best_params, test_accuracy, final_loss, results_df):
    # Generate report.md
    with open(file_path, 'w') as f:
        # Write experiment results table
        f.write("## Experiment Results\n")
        f.write(results_df.to_markdown(index=False) + "\n\n")

        # Write best parameters and performance
        f.write("## Best Parameters and Performance\n")
        f.write(f"- **Best Parameters**: {best_params}\n")
        f.write(f"- **Validation Accuracy**: {best_accuracy:.4f}\n")
        f.write(f"- **Test Accuracy**: {test_accuracy:.4f}\n")
        f.write(f"- **Final Loss**: {final_loss:.4f}\n")


# Main function
if __name__ == "__main__":
    parent_path = './resource'  # Dataset root directory
    train_tsv_path = os.path.join(parent_path, 'gt-train.tsv')
    X, y = get_dataset(parent_path, file_path=train_tsv_path, num_samples=None)

    # split train and test
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Perform hyperparameter tuning
    results_df, best_params, best_accuracy = hyperparameter_tuning(X_train, y_train, X_val, y_val)

    # Train the final model with the best parameters

    # Get test data
    test_tsv_path = os.path.join(parent_path, 'gt-test.tsv')
    X_test, y_test = get_dataset(parent_path, file_path=test_tsv_path, num_samples=None)

    final_model = MLPClassifier(
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        learning_rate_init=best_params['learning_rate'],
        max_iter=1000,  # Increase iterations
        random_state=42
    )
    final_model.fit(X_test, y_test)

    # Get final loss and test accuracy
    final_loss = final_model.loss_
    test_accuracy = final_model.score(X_test, y_test)
    print(f'Final Loss: {final_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # Generate report.md
    with open('./resource/report.md', 'w') as f:
        # Write experiment results table
        f.write("## Experiment Results\n")
        f.write(results_df.to_markdown(index=False) + "\n\n")

        # Write best parameters and performance
        f.write("## Best Parameters and Performance\n")
        f.write(f"- **Best Parameters**: {best_params}\n")
        f.write(f"- **Validation Accuracy**: {best_accuracy:.4f}\n")
        f.write(f"- **Test Accuracy**: {test_accuracy:.4f}\n")
        f.write(f"- **Final Loss**: {final_loss:.4f}\n")
    # report_result('./resource/report.md', best_accuracy, best_params, test_accuracy, final_loss, results_df)