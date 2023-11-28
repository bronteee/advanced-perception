import xgboost as xgb
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from dask_ml.model_selection import RandomizedSearchCV
import lightgbm as lgb
import numpy as np
import re
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# Load the large CSV file using Dask
ddf = dd.read_csv('/notebooks/advanced-perception/stock-closing.nosync/data/train_added_features.csv')

ddf = ddf.persist()

print("loaded dataset")

# Assuming the target variable is in the 'target' column
target_column = 'target'
X = ddf.drop(target_column, axis=1)
y = ddf[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print("split dataset")

#------------------------------ Light GBM model -------------------------------------------

# Replace special characters in column names with underscores
X_train.columns = [re.sub(r'\W+', '_', col) for col in X_train.columns]
X_test.columns = [re.sub(r'\W+', '_', col) for col in X_test.columns]

# LightGBM dataset
train_data = lgb.Dataset(X_train.compute(), label=y_train.compute())
val_data = lgb.Dataset(X_test.compute(), label=y_test.compute(), reference=train_data)

print("split into train and val")
# Set parameters for LightGBM
lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 50,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'lambda_l1': 1,  # Regularization parameter
    'bagging_freq': 5,
    'verbose': 1,
}

# Train the LightGBM model
num_round = 4000  # You can adjust this based on your dataset
bst = lgb.train(lgb_params, train_data, num_round, valid_sets=[val_data])
# Make predictions on the validation set
y_pred = bst.predict(X_test.compute(), num_iteration=bst.best_iteration)

# Calculate MAE on the validation set
mae = mean_absolute_error(y_test.compute(), y_pred)
print(f'Mean Absolute Error on Validation Set with LGBM: {mae}')
