from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from datetime import datetime
from numpy import absolute
from dataset import DATA_FILE_DIR
from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV

save_dir = "/notebooks/advanced-perception/stock-closing.nosync/models/"
seed = 42

train_data = pd.read_csv(DATA_FILE_DIR)

param_grid = {
    "max_depth": [3, 5],
    "learning_rate": [0.03, 0.05],
    "n_estimators": [50, 100],
    "colsample_bytree": [0.5, 0.7],
    "subsample": [0.3, 0.5],
}
grid_search = GridSearchCV(
    estimator=XGBRegressor(seed),
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    cv=RepeatedKFold(n_splits=10, n_repeats=1, random_state=seed),
    verbose=2,
)

X = train_data.drop(columns=["target"])
y = train_data["target"]

grid_search.fit(X, y)

model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

# If saved model exists, load it
try:
    model.load_model(f"{save_dir}/xgb_model.json")
    print(f"Model loaded from {save_dir}.")
except:
    print("Fitting model...")
    model.fit(X, y)
    # Save model
    model.save_model(f"{save_dir}/xgb_model.json")
    print(f"Model saved to {save_dir}.")

# Define cross-validation
print("Evaluating model with cross validation...")
cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=seed)

# Evaluate model
scores = cross_val_score(
    model, X, y, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1, verbose=2
)
scores = absolute(scores)
print(f'Mean MAE: {np.mean(scores):.3f} ({np.std(scores):.3f})')
