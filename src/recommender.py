import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def recommend_neighborhood(user_input, model, df, X, scaler, imputer):
    # Preprocess user input
    user_features = np.array([[user_input['budget'], user_input['sq_mtrs'], user_input['bedrooms'], user_input['bathrooms']]])
    user_features_imputed = imputer.transform(user_features)
    user_features_scaled = scaler.transform(user_features_imputed)
    
    # Add zero-padding for neighborhood features
    num_neighborhood_features = X.shape[1] - 4
    user_features_padded = np.hstack((user_features_scaled, np.zeros((1, num_neighborhood_features))))
    
    # Predict cluster for user input
    user_cluster = model.predict(user_features_padded)[0]
    
    # Get indices of properties in the same cluster
    cluster_indices = np.where(model.labels_ == user_cluster)[0]
    
    # Calculate distances to all properties in the cluster
    distances = euclidean_distances(user_features_scaled, X[cluster_indices, :4])[0]
    
    # Get top 5 closest properties
    top_5_indices = cluster_indices[np.argsort(distances)[:5]]
    
    # Get recommended neighborhoods
    recommended_neighborhoods = df.iloc[top_5_indices]['Neighborhood'].tolist()
    recommended_properties = df.iloc[top_5_indices].to_dict('records')
    
    return recommended_neighborhoods, recommended_properties