from flask import Flask, render_template, request
from markupsafe import Markup
from src.data_preprocessing import prepare_data
from src.model import train_model, find_optimal_clusters
from src.recommender import recommend_neighborhood
from src.utils import prepare_results
import ollama
import requests
import feedparser
from urllib.parse import quote
import time
import googlemaps
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Google Maps client
gmaps = googlemaps.Client(key=os.getenv('GOOGLE_MAPS_API_KEY'))

# Load and preprocess data
df, X, scaler, imputer = prepare_data('data/kenya_rentals.csv')

# Find optimal number of clusters
optimal_clusters = find_optimal_clusters(X)

# Train the model
model = train_model(X, n_clusters=optimal_clusters)

def get_location_coordinates(address, neighborhood):
    """Get latitude and longitude for a given address"""
    try:
        # Append neighborhood and Kenya for more accurate results
        full_address = f"{address}, {neighborhood}, Kenya"
        geocode_result = gmaps.geocode(full_address)
        
        if geocode_result:
            location = geocode_result[0]['geometry']['location']
            return location['lat'], location['lng']
        return None
    except Exception as e:
        print(f"Error getting coordinates: {e}")
        return None

def search_nearby_places(location, place_type, radius=2000, max_results=3):
    """Search for nearby places of a specific type"""
    try:
        places_result = gmaps.places_nearby(
            location=location,
            radius=radius,
            type=place_type
        )
        
        places = []
        if places_result.get('results'):
            for place in places_result['results'][:max_results]:
                place_details = {
                    'name': place.get('name'),
                    'address': place.get('vicinity'),
                    'rating': place.get('rating', 'N/A'),
                    'distance': radius  # You might want to calculate actual distance
                }
                places.append(place_details)
        
        return places
    except Exception as e:
        print(f"Error searching nearby places: {e}")
        return []

def get_amenities_info(neighborhood, address):
    """Get amenities information using Google Maps API and format with Ollama"""
    # Define amenity types to search for
    amenity_types = {
        'schools': 'school',
        'hospitals': 'hospital',
        'restaurants': 'restaurant',
        'supermarkets': 'supermarket',
        'banks': 'bank'
    }
    
    # Get coordinates for the location
    coordinates = get_location_coordinates(address, neighborhood)
    if not coordinates:
        return "Unable to find location coordinates."
    
    # Collect amenities information
    amenities_data = {}
    for amenity_name, amenity_type in amenity_types.items():
        places = search_nearby_places(coordinates, amenity_type)
        amenities_data[amenity_name] = places
    
    # Create a structured text of amenities for Ollama
    amenities_text = f"Here are the amenities near {address} in {neighborhood}:\n\n"
    for amenity_type, places in amenities_data.items():
        amenities_text += f"{amenity_type.title()}:\n"
        for place in places:
            amenities_text += f"- {place['name']} (Rating: {place['rating']})\n"
        amenities_text += "\n"
    
    # Use Ollama to create a natural description
    prompt = f"""Convert this amenities information into a natural, flowing paragraph that a real estate agent might use (only respond with the agent's part not your own intro e.g "Here's a natural-sounding paragraph that a real estate agent might use to highlight the amenities near the property:" no):

{amenities_text}

Focus on highlighting the convenience and accessibility of the location."""

    response = ollama.chat(model='tinyllama', messages=[
        {'role': 'user', 'content': prompt},
    ])
    
    return response['message']['content']

def get_neighborhood_news(neighborhood, num_news=2):
    """Get recent news about neighborhood using Google News RSS"""
    try:
        encoded_query = quote(f"{neighborhood} Kenya news")
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        feed = feedparser.parse(response.content)
        
        news_items = []
        for entry in feed.entries[:num_news]:
            news_items.append({
                'title': entry.title,
                'link': entry.link,
                'date': ' '.join(entry.published.split()[1:5])
            })
        
        return news_items
        
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = {
            'budget': float(request.form['budget']),
            'bedrooms': int(request.form['bedrooms']),
            'bathrooms': int(request.form['bathrooms']),
            'sq_mtrs': float(request.form['sq_mtrs'])
        }
        
        recommended_neighborhoods, recommended_properties = recommend_neighborhood(
            user_input, model, df, X, scaler, imputer
        )
        
        formatted_properties = prepare_results(recommended_properties)
        
        # Fetch additional information for each property
        for property in formatted_properties:
            property['amenities'] = get_amenities_info(
                property['Neighborhood'], 
                property.get('Address', f"{property['Neighborhood']}, Nairobi")  # Fallback address
            )
            property['news'] = get_neighborhood_news(property['Neighborhood'])
            
            # Add a small delay to prevent overwhelming external services
            time.sleep(1)
        
        return render_template(
            'results.html',
            neighborhoods=recommended_neighborhoods,
            properties=formatted_properties
        )
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)