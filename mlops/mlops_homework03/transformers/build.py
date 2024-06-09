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

    categorical = ['PULocationID', 'DOLocationID']
    train_dicts = df[categorical].to_dict(orient='records')
    
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    
    target = 'duration'
    y_train = df[target].values
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_train)

    print(f'{lr.intercept_}')
    
    return lr, dv