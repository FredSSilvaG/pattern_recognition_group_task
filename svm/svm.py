from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV,cross_val_score, KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import hinge_loss, accuracy_score, classification_report

import utils
import importlib
importlib.reload(utils)


def svm() -> None:
    args = utils.parse_args()

    X_train, y_train, X_test, y_test = utils.load_mnist_data()
    print("load success:", X_train.shape)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    pca = PCA(n_components=50)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    param_grid = {
    'C': [10],  
    'kernel': ['linear','rbf'], 
    'gamma': ['scale']  
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    report_results = []
    row_names = []
    svm = SVC(cache_size=10000)
    # for kernel in param_grid['kernel']:
    #     for C in param_grid['C']:
    #         report_results_K=[]

    #         row_names.append(kernel)
    #         report_results_K.append(float(C))

    #         svm = SVC(cache_size=10000,kernel=kernel,C=C,gamma='scale')
    #         # Compute cross-validated accuracy
    #         accuracy_scores = cross_val_score(svm, X_train, y_train, cv=kf, scoring='accuracy')
    #         mean_accuracy = np.mean(accuracy_scores)
    #         std_accuracy = np.std(accuracy_scores)

    #         report_results_K.append(float(mean_accuracy))
    #         report_results_K.append(float(std_accuracy))

    #         # Compute cross-validated hinge loss
    #         hinge_losses = []
    #         for train_idx, val_idx in kf.split(X_train):
    #              X_train_split, X_val = X_train[train_idx], X_train[val_idx]
    #              y_train_split, y_val = y_train[train_idx], y_train[val_idx]
    #              svm.fit(X_train_split, y_train_split)
    #              y_pred = svm.decision_function(X_val)
    #              hinge_losses.append(hinge_loss(y_val, y_pred))

    #         mean_loss = np.mean(hinge_losses)  
    #         std_loss = np.std(hinge_losses)
    #         report_results_K.append(float(mean_loss))
    #         report_results_K.append(float(std_loss))
    #         report_results.append(report_results_K)
    #         print(f"kernel={kernel}: C={C}, Accuracy = {mean_accuracy:.4f} (±{std_accuracy:.4f}), Hinge Loss = {mean_loss:.4f} (±{std_loss:.4f})")


    col_names = ["C","Mean Accuracy","Std Accuracy","Mean Hinge Loss","Std Hinge Loss"]
    

    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs = -1)
    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)

    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_best)
    print(f"Best SVM accuracy on test set: {accuracy:.4f}")
    best_model_classification_report = classification_report(y_test, y_pred_best)

    # Each result is added to the write
    utils.save_markdown_report(
        args.out_dir / "report.md",
        report_results,
        row_names=row_names,
        col_names=col_names,
        best_model = grid_search.best_params_,
        best_model_accuracy = f"{100 * accuracy:.2f}%",
        best_model_classification_report = best_model_classification_report
    )


if __name__ == "__main__":
    svm()