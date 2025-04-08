from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

import utils



def svm() -> None:
    X_train, y_train, X_test, y_test = utils.load_mnist_data()

    X_train, y_train = X_train[:10000], y_train[:10000]

    pca = PCA(n_components=100)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    param_grid = {
    'C': [0.1, 1, 10],  
    'kernel': ['linear'], 
    'gamma': ['scale', 'auto']  
    }
    print(param_grid['C'])
    
    svm = SVC(cache_size=1000)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs = 3)
    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)

    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)

    print(classification_report(y_test, y_pred_best))


if __name__ == "__main__":
    svm()