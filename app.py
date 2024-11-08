from flask import Flask, render_template, request
from src.data_preprocessing import prepare_data
from src.model import train_model, find_optimal_clusters
from src.recommender import recommend_neighborhood
from src.utils import prepare_results
import ollama

app = Flask(__name__)

# Load and preprocess data
df, X, scaler, imputer = prepare_data('data/kenya_rentals.csv')

# Find optimal number of clusters
optimal_clusters = find_optimal_clusters(X)

# Train the model
model = train_model(X, n_clusters=optimal_clusters)

def get_amenities_info(neighborhood, address):
    prompt = f"List the main amenities (schools, restaurants, hospitals) near {address} in {neighborhood}, Kenya. Include approximate distances."
    response = ollama.chat(model='tinyllama', messages=[
        {'role': 'user', 'content': prompt},
    ])
    return response['message']['content']

def get_neighborhood_news(neighborhood):
    prompt = f"Summarize the latest news about {neighborhood}, Kenya in 2-3 sentences."
    response = ollama.chat(model='tinyllama', messages=[
        {'role': 'user', 'content': prompt},
    ])
    return response['message']['content']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = {
            'budget': float(request.form['budget']),
            'bedrooms': int(request.form['bedrooms']),
            'bathrooms': int(request.form['bathrooms']),
            'sq_mtrs': float(request.form['sq_mtrs'])
        }
        recommended_neighborhoods, recommended_properties = recommend_neighborhood(user_input, model, df, X, scaler, imputer)
        
        formatted_properties = prepare_results(recommended_properties)
        
        # Fetch additional information for each property
        for property in formatted_properties:
            property['amenities'] = get_amenities_info(property['Neighborhood'], property.get('Address', 'Unknown'))
            property['news'] = get_neighborhood_news(property['Neighborhood'])
        
        return render_template('results.html', neighborhoods=recommended_neighborhoods, properties=formatted_properties)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)