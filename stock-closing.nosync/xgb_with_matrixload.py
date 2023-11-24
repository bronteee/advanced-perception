import xgboost as xgb
import pandas as pd
from dataset import DATA_FILE_DIR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the large CSV file
df = pd.read_csv(DATA_FILE_DIR)
df = df.fillna(df.mean())

# Assuming the target variable is in the 'target' column
target_column = 'target'
X = df.drop(target_column, axis=1)
y = df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DMatrix for training and testing
dtrain = xgb.DMatrix(X_train, label=y_train)
# Specify the parameter grid for RandomizedSearchCV
param_dist = {
    'objective': ['reg:squarederror'],
    'eval_metric': ['mae'],
    'max_depth': [3, 6, 9],
    'eta': [0.1, 0.2, 0.3],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'num_rounds': [50, 100, 150],
}

# Create an XGBoost regressor
xgb_reg = xgb.XGBRegressor()

# Use RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    xgb_reg, param_distributions=param_dist, n_iter=10, scoring='neg_mean_absolute_error', cv=3, verbose=1, n_jobs=-1
)

# Fit the model to the training data
random_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = random_search.best_params_

# Train the XGBoost model with the best hyperparameters
best_model = xgb.train(best_params, dtrain, best_params['num_rounds'])

# Make predictions on the test set
dtest = xgb.DMatrix(X_test)
y_pred = best_model.predict(dtest)

# Evaluate the model using Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error with Hyperparameter Tuning: {mae}')
