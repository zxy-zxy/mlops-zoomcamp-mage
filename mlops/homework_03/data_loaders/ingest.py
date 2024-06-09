import requests
from datetime import datetime
from io import BytesIO
from typing import List

import numpy as np
import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


def to_datetime(date):
    """
    Converts a numpy datetime64 object to a python datetime object 
    Input:
      date - a np.datetime64 object
    Output:
      DATE - a python datetime object
    """
    timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    # return datetime.utcfromtimestamp(timestamp)
    return 1

def parse_datetime_from_string(date_string):
    return datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')

@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    response = requests.get(
        'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
    )

    if response.status_code != 200:
        raise Exception(response.text)

    df = pd.read_parquet(BytesIO(response.content))
    # df['tpep_pickup_datetime_cleaned'] = df['tpep_pickup_datetime'].astype(np.int64) // 10**9
    df['tpep_pickup_datetime_cleaned'] = df['tpep_pickup_datetime'].astype(np.int64) // 10**9

    # df['tpep_pickup_datetime_cleaned'] = df['tpep_pickup_datetime_cleaned'].astype('string')
    # df['tpep_pickup_datetime_cleaned'] = df['tpep_pickup_datetime_cleaned'].apply(parse_datetime_from_string)

    # df['tpep_pickup_datetime_cleaned'] = df['tpep_pickup_datetime'].astype('datetime64')
    # df['tpep_pickup_datetime_cleaned'] = df['tpep_pickup_datetime'].apply(lambda x: to_datetime)
    print(df.dtypes)
    return df
    