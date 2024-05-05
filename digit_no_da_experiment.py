""" File to test the models on the digit dataset
- preprocessed data are available in data/digit.pkl file
- a model is trained on the preprocessed data
- Hyperparameters are tuned using GridSearchCV
- Model is evaluated on the test data
"""

import pickle
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


# Load the preprocessed data
with open('data/digit.pkl', 'rb') as f:
    data = pickle.load(f)
mnist = data['svhn']
X, y = mnist['X'], mnist['y']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
X_train = X_train[::5]
y_train = y_train[::5]
print(f"Train data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Create a pipeline
# pipe = make_pipeline(PCA(whiten=True), SVC(kernel='rbf', C=10, gamma=0.001))
pipe = make_pipeline(SVC(kernel='rbf', C=100, gamma=0.01))

# Perform GridSearchCV
param_grid = {
    'svc__C': [10, 100, 1000],
    'svc__gamma': [0.001, 0.01, 0.1]
}
grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best parameters: {grid.best_params_}")
print(f"Training accuracy: {grid.score(X_train, y_train)*100:.2f}%")
print(f"Test accuracy: {grid.score(X_test, y_test)*100:.2f}%")
