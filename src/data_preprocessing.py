import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Remove commas and 'KSh' from Price column and convert to float
    df['Price'] = df['Price'].str.replace('KSh ', '').str.replace(',', '').astype(float)
    
    # Convert sq_mtrs to numeric, replacing any non-numeric values with NaN
    df['sq_mtrs'] = pd.to_numeric(df['sq_mtrs'], errors='coerce')
    
    # Select relevant features
    features = ['Price', 'sq_mtrs', 'Bedrooms', 'Bathrooms']
    X = df[features]
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Normalize the features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_imputed)
    
    return df, X_normalized, scaler, imputer

def encode_neighborhood(df):
    # Encode neighborhood as categorical data
    return pd.get_dummies(df['Neighborhood'], prefix='Neighborhood')

def prepare_data(file_path):
    df, X_normalized, scaler, imputer = load_and_preprocess_data(file_path)
    neighborhood_encoded = encode_neighborhood(df)
    
    # Combine normalized features with encoded neighborhood
    X_final = np.hstack((X_normalized, neighborhood_encoded))
    
    return df, X_final, scaler, imputer