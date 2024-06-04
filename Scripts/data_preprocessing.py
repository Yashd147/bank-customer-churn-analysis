import pandas as pd
import numpy as np

def preprocess_data(data_file):
    data = pd.read_csv(data_file)

    # Data Preprocessing
    data['Geography'] = data['Geography'].replace({'Germany': 0, 'France': 1, 'Spain': 2})
    data['Gender'] = data['Gender'].replace({'Female': 0, 'Male': 1})
    data.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

    # Perform one-hot encoding for categorical variables
    data_encoded = pd.get_dummies(data)

    return data_encoded
