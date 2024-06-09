import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def train_model(df: pd.DataFrame, **kwargs):
    """
    Train a linear regression model using DictVectorizer on categorical features.
    Returns a dictionary containing the DictVectorizer and the trained model.
    """
    # Define categorical columns
    categorical = ['PULocationID', 'DOLocationID']
    
    # Convert categorical features to dictionary format
    train_dicts = df[categorical].to_dict(orient='records')
    
    # Apply DictVectorizer to transform categorical features into a feature matrix
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    
    # Define the target variable
    target = 'duration'
    y_train = df[target].values
    
    # Train a linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Make predictions on the training set
    y_pred = lr.predict(X_train)
    
    # Calculate the Root Mean Squared Error (RMSE) on the training set
    train_rmse = mean_squared_error(y_train, y_pred, squared=False)
    print(f'Train RMSE: {train_rmse}')
    
    # Print the intercept of the model
    print(f'Intercept of the model: {lr.intercept_}')
    
    # Return the DictVectorizer and the trained model
    return (lr, dv)